#### import from coconet
from pydca.contact_visualizer.contact_visualizer import DCAVisualizer
from pydca.fasta_reader import fasta_reader
from inputreader import InputReader

import subprocess
import numpy as np
import logging
import os, errno
import glob
from datetime import datetime
import pickle
import random
from pathlib import Path
from argparse import ArgumentParser
import sys

#### import from co-evolution transformer
from model.model import Model, ClassificationHead1, ClassificationHead2

#### essential packages
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from misc import *
from create_dataset import *

import time as time
import datetime
import copy

logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)

def main():
    parser = argparse.ArgumentParser("RNA Contact Prediction by Efficient Protein Transformer Transfer")
    parser.add_argument('--num_classes', default=37, type=int)
    parser.add_argument('--optim', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--total_epoch', default=100, type=int)
    parser.add_argument('--num_warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--feature_list', nargs='*', default=[0,1,2,3,4,5,6], type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--N', default=1, type=float)
    parser.add_argument('--model_type', default='parallel_model', type=str, choices=['concat_model', 'parallel_model'])
    parser.add_argument('--scheduler_type', default='CosineLR', type=str, choices=['CosineLR', 'StepLR'])
    args = parser.parse_args()

    hparams_dict = dict()
    for arg in vars(args):
        hparams_dict[arg] = getattr(args, arg)
        print(arg, getattr(args, arg))

    args.eval_window = args.total_epoch // 100

    ####### Set random seed
    random.seed(234)
    np.random.seed(234)
    torch.manual_seed(234)
    torch.cuda.manual_seed_all(234)

    ####### Prepare the model used for training
    model = Model()    ##### init a "modified" model
    weight_path = os.path.join(os.path.dirname(__file__), "weights.chk")    ##### load the weights
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))    ##### get state dict (protein)
    model.load_state_dict(state_dict, strict=False)
    print("Co-evolution Transformer loaded!")

    model = model.cuda()

    ####### RNA TRAIN SET
    train_dataset = CoCoNetDataset(data_dir='RNA_DATASET/')
    train_rna_lens = train_dataset.get_refseqs_len()
    train_pdb_data_pickle_file = 'RNA_DATASET_PDB_DATA.pickle'
    if os.path.exists(train_pdb_data_pickle_file):
        with open(train_pdb_data_pickle_file, 'rb') as handle:
            train_pdb_data = pickle.load(handle)
    else:
        train_pdb_data = train_dataset.get_pdb_data()
        with open(train_pdb_data_pickle_file, 'wb') as handle:
            pickle.dump(train_pdb_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("RNA_DATASET loaded!")

    ####### RNA TEST SET
    test_dataset = CoCoNetDataset(data_dir='RNA_TESTSET/')
    test_rna_lens = test_dataset.get_refseqs_len()
    test_pdb_data_pickle_file = 'RNA_TESTSET_PDB_DATA.pickle'
    if os.path.exists(test_pdb_data_pickle_file):
        with open(test_pdb_data_pickle_file, 'rb') as handle:
            test_pdb_data = pickle.load(handle)
    else:
        test_pdb_data = test_dataset.get_pdb_data()
        with open(test_pdb_data_pickle_file, 'wb') as handle:
            pickle.dump(test_pdb_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("RNA_TESTSET loaded!")

    ####### RNA Training dataset in Tensors (using off-the-shelf excutable)
    train_pdb_feat_pickle_file = 'RNA_DATASET_PDB_FEATS.pickle'
    if os.path.exists(train_pdb_feat_pickle_file):
        with open(train_pdb_feat_pickle_file, 'rb') as handle:
            train_input_tensors = pickle.load(handle)
    else:
        train_input_tensors = dict()
        for rna_fam_name, msa_file in tqdm(zip(train_dataset.msa_file_names_list, train_dataset.msa_files_list)):
            assert rna_fam_name in msa_file
            feat = get_a3m_feat(msa_file)
            with torch.no_grad():
                model.eval()
                feats = model(feat, return_feats=True)
                train_input_tensors[rna_fam_name] = [feat.cpu() for feat in feats]
        print("features of training MSAs extracted by get_a3m_feat()!")
        with open(train_pdb_feat_pickle_file, 'wb') as handle:
            pickle.dump(train_input_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("RNA_DATASET FEAT loaded!")

    test_pdb_feat_pickle_file = 'RNA_TESTSET_PDB_FEATS.pickle'
    if os.path.exists(test_pdb_feat_pickle_file):
        with open(test_pdb_feat_pickle_file, 'rb') as handle:
            test_input_tensors = pickle.load(handle)
    else:
        test_input_tensors = dict()
        for rna_fam_name, msa_file in tqdm(zip(test_dataset.msa_file_names_list, test_dataset.msa_files_list)):
            assert rna_fam_name in msa_file
            feat = get_a3m_feat(msa_file)
            with torch.no_grad():
                model.eval()
                feats = model(feat, return_feats=True)
                test_input_tensors[rna_fam_name] = [feat.cpu() for feat in feats]
        print("features of testing MSAs extracted by get_a3m_feat()!")
        with open(test_pdb_feat_pickle_file, 'wb') as handle:
            pickle.dump(test_input_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("RNA_TESTSET FEAT loaded!")

    ####### Sequences that are too long
    del train_input_tensors['RF02540']
    del train_pdb_data['RF02540']

    names = [name for name in train_input_tensors]
    for name in names:
        feats = train_input_tensors[name]
        feats = [feats[idx*2] for idx in args.feature_list]
        ##### normalization
        feats = torch.cat(feats, dim=-1)
        assert feats.shape[1] == feats.shape[2]
        B, L, _, C = feats.shape
        feats = feats.reshape(B, -1, C)
        feats = torch.nn.functional.normalize(feats, dim=1)
        # feats = torch.randn(feats.shape)      ##### for ablation study
        train_input_tensors[name] = feats.reshape(B, L, L, C)

        ##### train_input_tensors['RF00001'] = torch.randn(1, 120, 120, 96 x 1)

    ####### The training takes 29GB VRAM, requiring RTX-A6000
    train_label_tensor = dict()
    for rna_fam_name in train_pdb_data:
        rna_seq_len = train_rna_lens[rna_fam_name]     ##### length of this RNA
        train_label = np.ones((rna_seq_len, rna_seq_len)) * -100    ##### ignore index is -100
        for k, v in train_pdb_data[rna_fam_name].items():
            i, j = k[0], k[1]
            if abs(i-j) > 4:
                lbl = distance_to_37(v[-1]) if args.num_classes==37 else distance_to_2(v[-1])
                train_label[i, j] = lbl
                train_label[j, i] = lbl
        train_label = torch.from_numpy(train_label).long().unsqueeze(0)
        train_label_tensor[rna_fam_name] = train_label     ##### train_label_tensor['RF00001'] = torch.long(L, L, 1)
    print("labels pre-processed: L x L matrix with 37/2 classes!")

    ######################################
    #########  Validation Set  ###########
    ######################################'
    val_input_tensors  = dict()
    val_label_tensor = dict()

    val_names = ['RF01510', 'RF01689', 'RF01725', 'RF01734', 'RF01750', 'RF01763', 'RF01767', 'RF01786', 'RF01807']    #### val_1
    # val_names = ['RF01826', 'RF01831_1', 'RF01852', 'RF01854', 'RF01982', 'RF02001_2', 'RF02266', 'RF02447', 'RF02553']
    # val_names = ['RF00442_1', 'RF00458', 'RF00504', 'RF00606_1', 'RF01750', 'RF01763', 'RF01767', 'RF01786', 'RF01807']
    # val_names = ['RF00921', 'RF01051', 'RF01054', 'RF01300', 'RF01415', 'RF01510', 'RF01689', 'RF01725', 'RF01734']
    for name in val_names:
        val_input_tensors[name] = train_input_tensors[name]
        val_label_tensor[name] = train_label_tensor[name]
        #### then remove those RNAs from training set
        del train_input_tensors[name]
        del train_label_tensor[name]

    ####### PyTorch Dataset and DataLoader
    val_dataset = RNA_FEATSET(input_dict=val_input_tensors, label_dict=val_label_tensor)
    print('number of testing examples: ', len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(
                                                   dataset=val_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   collate_fn=pad_collate,
                                                  )

    ######################################
    #########  Validation Set  ###########
    ######################################

    ####### PyTorch Dataset and DataLoader
    train_dataset = RNA_FEATSET(input_dict=train_input_tensors, label_dict=train_label_tensor)
    print('number of training examples: ', len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
                                                   dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   collate_fn=pad_collate,
                                                  )

    ####### The testing RNAs
    names = [name for name in test_input_tensors]
    for name in names:
        # if name in ['RF00017', 'RF00023', 'RF00027', 'RF00102', 'RF00174', 'RF01073', 'RF01998', 'RF02012']:
        #     del test_input_tensors[name]
        #     del test_pdb_data[name]

        if name in ['RF01998', 'RF02012']:
            del test_input_tensors[name]
            del test_pdb_data[name]
        else:
            feats = test_input_tensors[name]
            feats = [feats[idx*2] for idx in args.feature_list]
            ##### normalization
            feats = torch.cat(feats, dim=-1)
            assert feats.shape[1] == feats.shape[2]
            B, L, _, C = feats.shape
            feats = feats.reshape(B, -1, C)
            feats = torch.nn.functional.normalize(feats, dim=1)
            test_input_tensors[name] = feats.reshape(B, L, L, C)

    test_label_tensor = dict()
    for rna_fam_name in test_pdb_data:
        rna_seq_len = test_rna_lens[rna_fam_name]     ##### length of this RNA
        test_label = np.ones((rna_seq_len, rna_seq_len)) * -100    ##### ignore index is -100
        for k, v in test_pdb_data[rna_fam_name].items():
            i, j = k[0], k[1]
            if abs(i-j) > 4:
                lbl = distance_to_37(v[-1]) if args.num_classes==37 else distance_to_2(v[-1])
                test_label[i, j] = lbl
                test_label[j, i] = lbl
        test_label = torch.from_numpy(test_label).long().unsqueeze(0)
        test_label_tensor[rna_fam_name] = test_label
    print("labels pre-processed: L x L matrix with 37/2 classes!")

    ####### PyTorch Dataset and DataLoader
    test_dataset = RNA_FEATSET(input_dict=test_input_tensors, label_dict=test_label_tensor)
    print('number of testing examples: ', len(test_dataset))
    test_dataloader = torch.utils.data.DataLoader(
                                                   dataset=test_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   collate_fn=pad_collate,
                                                  )

    ####### Learning partial parameters
    ClassificationHead = ClassificationHead1 if args.model_type == 'parallel_model' else ClassificationHead2
    learnable_model = ClassificationHead(num_feats=len(args.feature_list), num_classes=args.num_classes, kernel_size=args.kernel_size, bias=False)
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(learnable_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(learnable_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(learnable_model.parameters(), lr=args.lr, weight_decay=0.0001)


    ####### Cosine learning rate scheduler with warmup
    args.t_total = int(len(train_dataloader) * args.total_epoch)
    if args.scheduler_type == 'CosineLR':
        # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.t_total)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.t_total, eta_min=args.min_lr)
    else:
        milestone1 = int(args.t_total * 0.5)
        milestone2 = int(args.t_total * 0.75)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[milestone1, milestone2], gamma=0.5)

    ####### Training epochs
    cudnn.benchmark = True
    best_val = float('inf')

    print("Training starts!!!")
    # current_time = '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now())
    current_time = "{scheduler_type}-{lr}-{min_lr}-{bsz}-{epoch}".format(scheduler_type=args.scheduler_type, lr=args.lr, min_lr=args.min_lr, bsz=args.batch_size, epoch=args.total_epoch)
    print(current_time)
    writer = SummaryWriter(os.path.join('tensorboard_outdir/val/',  current_time))

    ####### Moving everything onto GPU
    model = model.cuda()
    learnable_model = torch.nn.DataParallel(learnable_model)
    learnable_model = learnable_model.cuda()

    label_weight = torch.tensor([0.1]*args.num_classes)
    if args.num_classes==37:
        label_weight[:16] = 10.0
    else:
        label_weight[:1] = 10.0
    criterion = torch.nn.CrossEntropyLoss(weight=label_weight, ignore_index=-100).cuda()

    best_ppv = 0

    for epoch in tqdm(range(args.total_epoch)):

        ##### set the model to training mode
        model.train()
        learnable_model.train()
        tr_loss = 0.0       #### training loss

        for i, data in enumerate(train_dataloader, 0):

            ###### input and label data
            input, lbl = data
            input = input.permute(0,3,1,2).cuda()
            lbl = lbl.cuda()

            ###### forward pass
            out = learnable_model(input)
            out = out + torch.transpose(out, dim0=2, dim1=3)      ##### make the prediction symmetric
            out = out.permute(0,2,3,1)
            loss = criterion(out.reshape(-1, args.num_classes), lbl.reshape(-1))
            tr_loss += loss.item()

            ###### backward pass
            loss.backward()

            ###### update parameters
            torch.nn.utils.clip_grad_norm_(learnable_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_loss = tr_loss / len(train_dataloader)
        writer.add_scalar('loss at each epoch', avg_loss, epoch)       ###### save training loss to TensorBoard at each epoch
        if epoch % args.eval_window == 0:
            rna_fam_names = test_dataset.rna_fam_names
            val_ppv = evaluation('val', learnable_model, val_dataloader, writer, epoch, args, rna_fam_names)
            if val_ppv > best_ppv:
                if not os.path.exists('saved_models/train_val/'):
                    os.makedirs('saved_models/train_val/')
                best_ppv = val_ppv
                test_ppv = evaluation('test', learnable_model, test_dataloader, writer, epoch, args, rna_fam_names)
                state_dict = copy.deepcopy(learnable_model.state_dict())
                torch.save(state_dict, 'saved_models/train_val/%s.chk'%current_time)     ##### save the model

    learnable_model.load_state_dict(state_dict, strict=True)
    args.N = 1
    ppv = evaluation('test', learnable_model, test_dataloader, writer, epoch, args, rna_fam_names)
    print("top L: ", ppv)
    args.N = 0.5
    ppv = evaluation('test', learnable_model, test_dataloader, writer, epoch, args, rna_fam_names)
    print("top 0.5L: ", ppv)
    args.N = 0.3
    ppv = evaluation('test', learnable_model, test_dataloader, writer, epoch, args, rna_fam_names)
    print("top 0.3L: ", ppv)
    print("Training ends!!!")
    print("Best testing ppv based on validation: ", test_ppv)


def evaluation(dataset_type, learnable_model, test_dataloader, writer, epoch, args, rna_fam_names):
    learnable_model.eval()

    avg_ppv = 0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader, 0):

            ###### input and label data
            input, lbl = data
            input = input.permute(0,3,1,2).cuda()
            lbl = lbl.cuda()

            ###### forward pass
            out = learnable_model(input)
            out = out + torch.transpose(out, dim0=2, dim1=3)      ##### make the prediction symmetric
            out = out.permute(0,2,3,1)
            out = torch.softmax(out, dim=-1)
            pred = torch.sum(out[..., :16], dim=-1)[0] if args.num_classes==37 else torch.sum(out[..., :1], dim=-1)[0]

            assert pred.shape[0] == pred.shape[1]
            L = pred.shape[0]
            for i in range(L):
                for j in range(L):
                    assert pred[i,j] == pred[j,i]    #### check if the prediction isi symmetric

            mask = torch.full((L, L), -10000)
            for i in range(L):
                for j in range(L):
                    if abs(i-j) > 4:
                        mask[i,j] = 0     #### mask has -10000 at diagonal
                    else:
                        pass

            pred = pred.cpu() + mask     #### mask out predictions at the digonal by setting |i-j|<=4 with very small (-10000) value
            delta = torch.randn(L,L) * 1e-7     #### add a tiny value (delta) to matrix, in case the prediction is degenerated
            pred = pred + delta + delta.T

            topk_values, _ = pred.reshape(-1).topk(k=int(2*args.N*L))    ###### because we use both upper and lower matrix for evaluation, pick top 2L predictions
            topk_value = topk_values[-1]     ##### the last value is the threshold value for top 2L prediction in the LxL matrix
            pred[pred<topk_value]  = -10000        ##### if the prediction is smaller than this threshold, clip to 0
            pred[pred>=topk_value] = 1                 ##### for other positions, they are predicted contacts
            pred[pred<=0] = 0

            grid = torchvision.utils.make_grid(pred)
            writer.add_image(rna_fam_names[idx], grid, epoch)

            lbl = lbl.cpu().squeeze(0) - mask     ##### lbl += 10000 for abs(i-j)<=4
            if args.num_classes==37:
                lbl[lbl<=-1]= 100
                lbl[lbl<16] = 1                       ##### lbl is 1 (contact) if distance is smaller than 10A, which corresponds to label 0,1,2,...,15
                lbl[lbl>=16]= 0                       ##### lbl is 0 (non-contact) if distance is larger than 10A, which corresponds to label 16,17,18,....
            else:
                lbl = -lbl
                lbl[lbl>=0] = 1
                lbl[lbl<0]= 0

            ppv = (pred * lbl).sum() / int(2*args.N*L)      ##### position-wise multiplication to find "positive prediction", divided by 2L (total number of predictions)
            avg_ppv += ppv.data

        avg_ppv /= len(test_dataloader)
        writer.add_scalar(dataset_type + ': PPV at each epoch', avg_ppv, epoch)
        return avg_ppv


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
