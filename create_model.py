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
from model.model import Model, ClassificationHead1
from collections import OrderedDict

#### essential packages
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from misc import *
from create_dataset import *

import time as time
import datetime

logger = logging.getLogger(__name__)
np.set_printoptions(threshold=sys.maxsize)


class CoT_RNA_Transfer(nn.Module):
    def __init__(self, backbone_path, transfer_path):
        super(CoT_RNA_Transfer, self).__init__()
        self.feature_list = [0, 1, 2, 3, 4, 5, 6]

        self.Backbone = Model()
        self.Backbone.eval()
        self.Transfer = ClassificationHead1(num_feats=len(self.feature_list), num_classes=37,
                                             kernel_size=3, bias=False)
        self.Transfer.eval()

        self.Backbone.load_state_dict(torch.load(backbone_path), strict=True)
        state_dict = torch.load(transfer_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        self.Transfer.load_state_dict(new_state_dict, strict=True)


    def forward(self, x):
        feat = self.Backbone(x, return_feats=True)
        feat = [feat[idx * 2] for idx in self.feature_list]
        feat = torch.cat(feat, dim=-1)

        assert feat.shape[1] == feat.shape[2]
        B, L, _, C = feat.shape
        feat = feat.reshape(B, -1, C)
        feat = torch.nn.functional.normalize(feat, dim=1)
        feat = feat.reshape(B, L, L, C)
        feat = feat.permute(0, 3, 1, 2)


        out = self.Transfer(feat)

        out = out + torch.transpose(out, dim0=2, dim1=3)  ##### make the prediction symmetric
        out = out.permute(0, 2, 3, 1)
        out = torch.softmax(out, dim=-1)
        pred = torch.sum(out[..., :16], dim=-1)[0]

        return pred, feat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='CoT_RNA_Transfer.chk', type=str)
    parser.add_argument('--backbone_path', default='./weights.chk', type=str)
    parser.add_argument('--transfer_path', default='saved_models/train_val/CosineLR-0.001-MIN_LR-BSZ-EPOCH.chk', type=str)
    args = parser.parse_args()

    hparams_dict = dict()
    for arg in vars(args):
        hparams_dict[arg] = getattr(args, arg)
        print(arg, getattr(args, arg))

    model = CoT_RNA_Transfer(backbone_path=args.backbone_path, transfer_path=args.transfer_path)

    # weight_path = os.path.join(os.path.dirname(__file__), args.model_name)
    # state_dict = torch.load(weight_path)
    # model.load_state_dict(state_dict)

    torch.save(model.state_dict(), args.model_name)
