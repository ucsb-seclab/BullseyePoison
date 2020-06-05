#!/usr/bin/env bash
python eval_poisons_transfer.py --gpu $1 --target-subset $2 --eval-poisons-root $3 --target-num $4 --target-label 1 --poison-label 6 --victim-net DPN92 SENet18 ResNet50 ResNeXt29_2x64d GoogLeNet MobileNetV2 ResNet18 DenseNet121
