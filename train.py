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
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fmnist_iid, fmnist_noniid
from utils.config import args
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.reshaper import flat_to_network, network_to_flat
from utils.compression import quantize, topk_sparsify

# ------------------------------------------------------------------------
# 1. Adjust device based on Gaudi vs. GPU
# ------------------------------------------------------------------------
if args.gaudi:
    # Make sure SynapseAI / habana-torch is installed
    import habana_frameworks.torch.core as htcore

    args.device = torch.device("hpu")
else:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu != -1 else "cpu")

print(args.device)


# ------------------------------------------------------------------------
# (Optional) A small helper to name your run properly
# ------------------------------------------------------------------------
def device_name(gaudi, eager, gpu):
    if gaudi:
        return 'HPU-Eager' if eager else 'HPU-Lazy'
    if gpu == -1:
        return 'CPU'
    return f'GPU_{gpu}'


# ------------------------------------------------------------------------
# (Optional) Set up profiling
# ------------------------------------------------------------------------
prof = None  # torch profiler object
hpu_profiler_started = False  # track if Habana profiler started

if getattr(args, 'profile', True):
    if args.gaudi:
        # HPU Profiler
        # Make sure you have installed habana_frameworks.torch.hpu
        # and set HABANA_PROFILE=1 if you want hardware-level traces
        from habana_frameworks.torch.hpu.profiler import start_profiler, stop_profiler

        start_profiler(
            output_dir="./habana_profile",  # Directory to store trace
            profile_type="hl"  # 'hl' = high-level, or 'hw' = hardware-level
        )
        hpu_profiler_started = True
        print("Habana profiler started.")
    else:
        # Torch Profiler for CPU/GPU
        import torch.profiler

        # Set up an on-trace callback to save events for TensorBoard
        # Adjust schedule to your preference
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ] if args.device.type == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        )
        prof.start()
        print("Torch profiler started.")

# ------------------------------------------------------------------------
# Main training script
# ------------------------------------------------------------------------
if __name__ == '__main__':
    wandb.init(project='gaudi-fl', config=args)
    config = wandb.config
    wandb.run.name = '{} s{}-{} -- {}'.format(
        args.dataset,
        args.seed,
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
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
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

    # (Optional) Eager mode for Gaudi
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

    for epoch in tqdm(range(args.epochs)):
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

        # --------------------------------------------------------------------
        # Compress & Average
        # --------------------------------------------------------------------
        w_flat = [network_to_flat(w) for w in w_locals]
        g_flat, meta = network_to_flat(w_glob, return_meta=True)
        grad_flat = [w - g_flat for w in w_flat]

        # Sparsify
        if args.s_rate < 1:
            grad_flat = [topk_sparsify(grad, args.s_rate) for grad in grad_flat]

        # Quantize
        if args.q_level < 32:
            grad_flat = [quantize(grad, args.q_level) for grad in grad_flat]

        w_flat = [g_flat + grad for grad in grad_flat]
        w_locals = [flat_to_network(w, meta) for w in w_flat]

        # FedAvg
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        # Logging and metrics
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

        # --------------------------------------------------------------------
        # If profiling with Torch Profiler on GPU/CPU, step each epoch
        # --------------------------------------------------------------------
        if prof is not None:
            prof.step()

    # ------------------------------------------------------------------------
    # Stop Profilers
    # ------------------------------------------------------------------------
    if prof is not None:
        prof.stop()
        print("Torch profiler stopped.")

    if hpu_profiler_started:
        from habana_frameworks.torch.hpu.profiler import stop_profiler

        stop_profiler()
        print("Habana profiler stopped.")

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
