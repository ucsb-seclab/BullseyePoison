#!/usr/bin/env bash
python eval_poisons_transfer.py --gpu $1 --target-index-start $2 --eval-poisons-root $3 --target-label 6 --poison-label 8 --target-net DPN92 SENet18 ResNet50 ResNeXt29_2x64d GoogLeNet MobileNetV2 ResNet18 DenseNet121--end2end True --retrain-lr 1e-4 --retrain-wd 5e-4 --retrain-epochs 60
