#!/usr/bin/env bash
python eval_poisons_transfer.py --gpu $1 --target-index-start $2 --eval-poisons-root $3 --target-label 6 --poison-label 8 --target-net DPN92 SENet18 ResNet50 ResNeXt29_2x64d GoogLeNet MobileNetV2 ResNet18 DenseNet121 --retrain-epochs 60 --test-chk-name cifar10-ckpt-%s-2400to1-dp0.000-droplayer0.000-seed2020.t7
