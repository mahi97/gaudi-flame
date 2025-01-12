#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import datetime
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from torchvision import datasets, transforms

# Import your sampling helpers, now extended with FMNIST and CIFAR non-IID
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fmnist_iid, fmnist_noniid

from utils.config import args
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# if args.eager:
#     os.environ['PT_HPU_LAZY_MODE'] = '0'

if args.gaudi:
    import habana_frameworks.torch.core as htcore
    args.device = torch.device("hpu")
else:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu != -1 else "cpu")

print(args.device)

def device_name(gaudi, eager, gpu):
    if gaudi:
        if eager:
            return 'HPU-Eager'
        return 'HPU-Lazy'
    if gpu == -1:
        return 'CPU'
    return 'GPU_' + str(gpu)

if __name__ == '__main__':

    wandb.init(project='gaudi-fl', config=args)
    config = wandb.config
    wandb.run.name = '{} s{}-{} -- {}'.format(args.dataset, args.seed,
                                             device_name(args.gaudi, args.eager, args.gpu),
                                             wandb.run.id)

    # ------------------------------------------------------------------------
    # Load dataset and split among users
    # ------------------------------------------------------------------------
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_train = datasets.FashionMNIST('./data/fmnist/', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('./data/fmnist/', train=False, download=True, transform=trans_fmnist)

        if args.iid:
            dict_users = fmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = fmnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            # Use the new CIFAR non-IID sampler
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # ------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        # Use the same CNNMnist class for FMNIST or roll your own if you want
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit("Error: Model not implemented for this dataset")

    print(net_glob)
    net_glob.train()
    if args.gaudi and args.eager:
        net_glob = torch.compile(net_glob, backend="hpu_backend")

    # copy global weights
    w_glob = net_glob.state_dict()

    # ------------------------------------------------------------------------
    # Federated training
    # ------------------------------------------------------------------------
    loss_train = []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for _ in range(args.num_users)]

    for epoch in range(args.epochs):
        loss_locals = []

        if not args.all_clients:
            w_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))

        # FedAvg
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        # Logging and metrics
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_train, _ = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        print(f'Epoch {epoch:3d}, Average loss {loss_avg:.3f}')
        loss_train.append(loss_avg)

        wandb.log({
            'round': epoch,
            'train_loss': loss_avg,
            'train_acc': acc_train,
            'test_acc': acc_test,
            'test_loss': loss_test
        })

    # ------------------------------------------------------------------------
    # Plot loss curve
    # ------------------------------------------------------------------------
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'./save/{timestamp}_fed_{args.dataset}_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}.png')

    # Final testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print(f"Final Training accuracy: {acc_train:.2f}")
    print(f"Final Testing accuracy:  {acc_test:.2f}")

    wandb.finish()
