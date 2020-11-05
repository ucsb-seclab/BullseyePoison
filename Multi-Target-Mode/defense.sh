#!/bin/bash
gpu=$1
method=$2
targetnum=$3
python defense.py --gpu $gpu --target-num $targetnum --eval-poisons-root attack-results/100-overlap/linear-transfer-learning/$method/1000 --target-net DPN92 SENet18 ResNet50 ResNeXt29_2x64d GoogLeNet MobileNetV2 ResNet18 DenseNet121 --target-label 1 --poison-label 6
