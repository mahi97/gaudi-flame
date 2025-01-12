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
from tqdm import tqdm

from torchvision import datasets, transforms

# Import your sampling helpers
from utils.sampling import (
    mnist_iid, mnist_noniid,
    cifar_iid, cifar_noniid,
    fmnist_iid, fmnist_noniid
)

from utils.config import args
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.reshaper import flat_to_network, network_to_flat
from utils.compression import quantize, topk_sparsify

# ------------------------------------------------------------------------
# 1. Set Device
# ------------------------------------------------------------------------
if args.gaudi:
    import habana_frameworks.torch.core as htcore
    args.device = torch.device("hpu")
else:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu != -1 else "cpu")

print(args.device)

def device_name(gaudi, eager, gpu):
    if gaudi:
        return 'HPU-Eager' if eager else 'HPU-Lazy'
    if gpu == -1:
        return 'CPU'
    return f'GPU_{gpu}'

# ------------------------------------------------------------------------
# 2. (Optional) Profiler Setup
# ------------------------------------------------------------------------
prof = None

if getattr(args, 'profile', True):
    import torch.profiler

    activities = [torch.profiler.ProfilerActivity.CPU]
    # If running on Gaudi, add HPU activity:
    if args.gaudi:
        activities.append(torch.profiler.ProfilerActivity.HPU)
    # Else if CUDA is available:
    elif args.device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=0,        # steps to wait before collecting
            warmup=0,     # warmup steps
            active=10,      # active steps (collected)
            repeat=1,
            skip_first=1
        ),
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    )
    prof.start()
    print("Torch profiler started with schedule: wait=0, warmup=0, active=10.")

# ------------------------------------------------------------------------
# Main Training
# ------------------------------------------------------------------------
if __name__ == '__main__':
    wandb.init(project='gaudi-fl', config=args)
    config = wandb.config
    wandb.run.name = '{} s{}-{} -- {}'.format(
        args.dataset, args.seed,
        device_name(args.gaudi, args.eager, args.gpu),
        wandb.run.id
    )

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

        dict_users = mnist_iid(dataset_train, args.num_users) if args.iid \
            else mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_train = datasets.FashionMNIST('./data/fmnist/', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('./data/fmnist/', train=False, download=True, transform=trans_fmnist)

        dict_users = fmnist_iid(dataset_train, args.num_users) if args.iid \
            else fmnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)

        dict_users = cifar_iid(dataset_train, args.num_users) if args.iid \
            else cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # ------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset in ['mnist', 'fmnist']):
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

    # Optional: Eager mode for Gaudi
    if args.gaudi and args.eager:
        net_glob = torch.compile(net_glob, backend="hpu_backend")

    w_glob = net_glob.state_dict()  # copy global weights

    loss_train = []
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for _ in range(args.num_users)]

    # ------------------------------------------------------------------------
    # Federated training
    # ------------------------------------------------------------------------
    for epoch in tqdm(range(args.epochs)):
        loss_locals = []

        if not args.all_clients:
            w_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            loss_locals.append(loss)

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            # Mark step for Gaudi right after local training
            if args.gaudi:
                htcore.mark_step()

        # --------------------------------------------------------------------
        # Compress & FedAvg
        # --------------------------------------------------------------------
        w_flat = [network_to_flat(w) for w in w_locals]
        g_flat, meta = network_to_flat(w_glob, return_meta=True)
        grad_flat = [wf - g_flat for wf in w_flat]

        # Sparsify
        if args.s_rate < 1:
            grad_flat = [topk_sparsify(grad, args.s_rate) for grad in grad_flat]

        # Quantize
        if args.q_level < 32:
            grad_flat = [quantize(grad, args.q_level) for grad in grad_flat]

        w_flat = [g_flat + gf for gf in grad_flat]
        w_locals = [flat_to_network(wf, meta) for wf in w_flat]

        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        # --------------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------------
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_train, _ = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        loss_train.append(loss_avg)

        wandb.log({
            'round': epoch,
            'train_loss': loss_avg,
            'train_acc': acc_train,
            'test_acc': acc_test,
            'test_loss': loss_test
        })

        # Step the profiler each epoch (for GPU or Gaudi)
        if prof is not None:
            prof.step()

    # ------------------------------------------------------------------------
    # Stop Profiler
    # ------------------------------------------------------------------------
    if prof is not None:
        prof.stop()
        print("Torch profiler stopped. Logs saved to ./profile_logs/")

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
    acc_train, loss_train_ = test_img(net_glob, dataset_train, args)
    acc_test, loss_test_ = test_img(net_glob, dataset_test, args)
    print(f"Final Training accuracy: {acc_train:.2f}")
    print(f"Final Testing accuracy:  {acc_test:.2f}")

    wandb.finish()
