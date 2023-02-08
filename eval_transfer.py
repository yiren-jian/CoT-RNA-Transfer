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

logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)

def main():
    parser = argparse.ArgumentParser("RNA Contact Prediction by Efficient Protein Transformer Transfer")
    parser.add_argument('--num_classes', default=37, type=int)
    parser.add_argument('--optim', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=56, type=int)
    parser.add_argument('--total_epoch', default=1000, type=int)
    parser.add_argument('--num_warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--feature_list', nargs='*', default=[0,1,2,3,4,5,6], type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--N', default=1, type=float)
    parser.add_argument('--model_type', default='parallel_model', type=str, choices=['concat_model', 'parallel_model'])
    parser.add_argument('--model_path', default='saved_models/train_val/CosineLR-0.001-0.0-4-500.chk', type=str)
    args = parser.parse_args()

    hparams_dict = dict()
    for arg in vars(args):
        hparams_dict[arg] = getattr(args, arg)
        print(arg, getattr(args, arg))

    args.eval_window = args.total_epoch // 100

    ####### Set random seed
    # random.seed(234)
    # np.random.seed(234)
    # torch.manual_seed(234)
    # torch.cuda.manual_seed_all(234)

    ####### Prepare the model used for training
    model = Model()    ##### init a "modified" model
    weight_path = os.path.join(os.path.dirname(__file__), "weights.chk")    ##### load the weights
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))    ##### get state dict (protein)
    model.load_state_dict(state_dict, strict=False)
    print("Co-evolution Transformer loaded!")

    model = model.cuda()

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

    test_pdb_feat_pickle_file = 'RNA_TESTSET_PDB_FEATS.pickle'
    if os.path.exists(test_pdb_feat_pickle_file):
        with open(test_pdb_feat_pickle_file, 'rb') as handle:
            test_input_tensors = pickle.load(handle)
    print("RNA_TESTSET FEAT loaded!")


    ####### The testing RNAs
    names = [name for name in test_input_tensors]
    for name in names:

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

    weight_path = os.path.join(os.path.dirname(__file__), args.model_path)    ##### load the weights
    state_dict = torch.load(weight_path)    ##### get state dict (protein)
    learnable_model = torch.nn.DataParallel(learnable_model)
    learnable_model = learnable_model.cuda()
    learnable_model.load_state_dict(state_dict, strict=True)
    print("learned model loaded!!!")

    rna_fam_names = test_dataset.rna_fam_names
    ppv = evaluation(learnable_model, test_dataloader, args, rna_fam_names)



def evaluation(learnable_model, test_dataloader, args, rna_fam_names):
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
            out = torch.softmax(out, dim=-1)  #### [1, L, L, 37]
            pred = torch.sum(out[..., :16], dim=-1)[0] if args.num_classes==37 else torch.sum(out[..., :1], dim=-1)[0]

            assert pred.shape[0] == pred.shape[1]
            L = pred.shape[0]
            for i in range(L):
                for j in range(L):
                    assert pred[i,j] == pred[j,i]    #### check if the prediction is symmetric

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
            print(rna_fam_names[idx], lbl.shape)
            print(rna_fam_names[idx], ppv.item())
            avg_ppv += ppv.data

        avg_ppv /= len(test_dataloader)
        print('avg PPV: ', avg_ppv.item())
        return avg_ppv


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
