#!/usr/bin/env bash
./docker_build_run.sh guadi-flame . ./data train.py --gaudi --dataset cifar10 --model cnn --iid --num_channels 3 --num_classes 10 --gpu 0 --epochs 100