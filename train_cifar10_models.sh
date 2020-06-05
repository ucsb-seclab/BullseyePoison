#!/usr/bin/env bash
for dp in 0.0
do
python train_cifar10_models.py --train-dp $dp --gpu $1 --net $2 --seed 2020 --sidx 1 --eidx 2400
done
