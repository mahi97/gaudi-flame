#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from typing import List, Any

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from utils.options import args

if args.gaudi:
    import habana_frameworks.torch.core as htcore

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    args: Any
    loss_func: nn.Module
    selected_clients: List[int]
    ldr_train: DataLoader

    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        net.train()
        if args.gaudi and args.eager:
            net = torch.compile(net, backend="hpu_backend")
        # train and update


        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                print('-1')
                optimizer.zero_grad()
                print('0')
                log_probs = net(images)
                print('1')
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                print('2')

                if self.args.gaudi and not self.args.eager:
                    htcore.mark_step()
                optimizer.step()
                if self.args.gaudi and not self.args.eager:
                    htcore.mark_step()
                print('3')

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

