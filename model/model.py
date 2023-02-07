# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from .base_model import BaseModel
from .msa_embeddings import MSAEmbeddings
from .attention import ZBlock, YBlock, YAggregator, ZRefiner
from .distance_predictor import DistancePredictor
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from .resnet import *

class CoT_RNA_Transfer(nn.Module):
    def __init__(self,):
        super(CoT_RNA_Transfer, self).__init__()
        self.feature_list = [0, 1, 2, 3, 4, 5, 6]

        self.Backbone = Model()
        self.Backbone.eval()
        self.Transfer = ClassificationHead1(num_feats=len(self.feature_list), num_classes=37,
                                             kernel_size=3, bias=False)
        self.Transfer.eval()



    def forward(self, x):
        feat = self.Backbone(x, return_feats=True)

        # with open('./RNA_TESTSET_PDB_FEATS.pickle', 'rb') as handle:
        #     test_input_tensors = pickle.load(handle)
        # feat_new = test_input_tensors['RF00001']

        feat = [feat[idx * 2] for idx in self.feature_list]
        # for i in range(len(feat)):
        #     feat_save = feat[i]
        #     print(feat_save.shape)
        #     print(torch.sum(feat_new[i * 2] - feat_save.cpu()))

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

class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.msa_embeddings = MSAEmbeddings(msa_gap=7, embed_dim=128, dropout=0.1)
        self.blocks = nn.ModuleList()
        nblocks = 6
        for i in range(nblocks):
            self.blocks.append(
                ZBlock(
                    ninp=128,
                    nhead=8,
                    dim2d=0 if i == 0 else 96,
                    rn_inp=96,
                    rn_layers=12,
                    dim_feedforward=256,
                    dropout=0.1,
                )
            )

            self.blocks.append(
                YBlock(ninp=128, nhead=4, dim_feedforward=256, dropout=0.1)
            )
        self.aggregator = YAggregator(ninp=128, nhead=32, nhid=8, dim2d=96, agg_dim=96)
        self.refiner = ZRefiner(ninp=96, repeats=12)

        self.distance_predictor = DistancePredictor(ninp=96)

    def forward(self, data, return_feats=False):
        x1d = self.msa_embeddings(data["seq"], data["msa"], data["index"])
        x2d = None
        feats = []
        for i, model_fn in enumerate(self.blocks):
            x1d, x2d = model_fn(x1d, x2d)
            feats.append(x2d.clone().detach())
        x2d = self.aggregator(x1d, x2d)
        x2d = self.refiner(x2d)
        feats.append(x2d.clone().detach())
        if not return_feats:
            return self.distance_predictor(x2d)
        else:
            return feats


class ClassificationHead1(nn.Module):   #### 'parallel_model'
    def __init__(self, num_feats=7, num_classes=37, kernel_size=3, bias=False):
        super(ClassificationHead1, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size == 3
        self.padding = 1

        self.num_feats = num_feats
        self.activation = nn.ReLU()

        self.channels = [96, 48, 24]

        self.parallel_models = nn.ModuleList()
        for i in range(self.num_feats):
            self.parallel_models.append(
                                        nn.Sequential(
                                            nn.Conv2d(in_channels=96, out_channels=self.channels[0], kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias),
                                            nn.BatchNorm2d(self.channels[0]),
                                            self.activation,
                                            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[0], kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias),
                                            nn.BatchNorm2d(self.channels[0]),
                                            self.activation,
                                            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias),
                                            nn.BatchNorm2d(self.channels[1]),
                                            self.activation,
                                            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias),
                                            nn.BatchNorm2d(self.channels[1]),
                                            self.activation,
                                        )
                                    )

        self.predictor = nn.Sequential(
                        nn.Conv2d(in_channels=self.channels[1]*self.num_feats, out_channels=self.channels[2]*self.num_feats, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias),
                        nn.BatchNorm2d(self.channels[2]*self.num_feats),
                        self.activation,
                        nn.Conv2d(in_channels=self.channels[2]*self.num_feats, out_channels=self.channels[2]*self.num_feats, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias),
                        nn.BatchNorm2d(self.channels[2]*self.num_feats),
                        self.activation,
                        nn.Conv2d(in_channels=self.channels[2]*self.num_feats, out_channels=num_classes, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias)
                    )

    def forward(self, x):
        xs = torch.split(x, split_size_or_sections=96, dim=1)
        assert self.num_feats == len(xs)

        ys = []
        for i in range(self.num_feats):
            y = self.parallel_models[i](xs[i])
            ys.append(y)
        ys = torch.cat(ys, dim=1)

        ys = self.predictor(ys)
        return ys


class ClassificationHead2(nn.Module):
    def __init__(self, num_feats=7, num_classes=37, kernel_size=3, bias=False):
        super(ClassificationHead2, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size == 3
        self.padding = 1

        self.num_feats = num_feats
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=96*self.num_feats // 1, out_channels=96*self.num_feats // 2, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(96*self.num_feats // 2)
        self.conv2 = nn.Conv2d(in_channels=96*self.num_feats // 2, out_channels=96*self.num_feats // 4, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(96*self.num_feats // 4)
        self.conv3 = nn.Conv2d(in_channels=96*self.num_feats // 4, out_channels=96*self.num_feats // 8, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias)
        self.bn3 = nn.BatchNorm2d(96*self.num_feats // 8)

        self.last = nn.Conv2d(in_channels=96*self.num_feats // 8, out_channels=self.num_classes, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=bias)

        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        return self.last(x)
