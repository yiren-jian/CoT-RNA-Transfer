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
from model.model import Model

#### essential packages
import argparse
import torch
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
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--total_epoch', default=1000, type=int)
    parser.add_argument('--num_warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
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
        pass

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
        train_label_tensor[rna_fam_name] = train_label
    print("labels pre-processed: L x L matrix with 37/2 classes!")

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
        if name in ['RF00017', 'RF00027', 'RF00102', 'RF00174', 'RF01998', 'RF02012']:
            del test_input_tensors[name]
            del test_pdb_data[name]

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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
