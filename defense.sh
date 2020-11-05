#!/bin/bash
python defense.py --gpu $1 --eval-poisons-root attack-results-25poisons-102flowers/100-overlap/linear-transfer-learning/mean/800 --model-resume-path model-chks --target-net DPN92 SENet18 ResNet50 ResNeXt29_2x64d GoogLeNet MobileNetV2 ResNet18 DenseNet121 --target-index-start 1 --target-index-end 11 --target-dset 102flowers --target-label -1
