import os
import json
import torch

# VICTIMS_LINE_STYLES = {'DPN92', 'SENet18', 'ResNet50', 'ResNeXt29_2x64d', 'GoogLeNet',
#                   'MobileNetV2', 'ResNet18', 'DenseNet121'}
VICTIMS_COLORS = {'DPN92': 'saddlebrown', 'SENet18': 'gray', 'ResNet50': 'seagreen', 'ResNeXt29_2x64d': 'violet',
                  'GoogLeNet': 'blue', 'MobileNetV2': 'gold', 'ResNet18': 'crimson', 'DenseNet121': 'cyan'}
VICTIMS_LINESTYLES = {'DPN92': 'solid', 'SENet18': 'solid', 'ResNet50': 'solid', 'ResNeXt29_2x64d': 'solid',
                  'GoogLeNet': 'solid', 'MobileNetV2': 'solid', 'ResNet18': 'dashed', 'DenseNet121': 'dashed'}

METHODS_LINESTYLES = {'mean': 'dashed', 'convex': 'solid', 'mean-3': 'dashdot', 'mean-5': 'dotted'}
METHODS_NAMES = {'mean': 'BP', 'convex': 'CP', 'mean-3': 'BP-3x', 'mean-5': 'BP-5x'}
METHODS_COLORS = {'mean': 'crimson', 'convex': 'gray', 'mean-3': 'darkcyan', 'mean-5': 'saddlebrown'}


def read_all_in_dir(poisons_root_path, retrain_epochs, target_num):
    filename = 'eval-retrained-for-{}epochs.json'.format(retrain_epochs)
    all_res = {'targets': {}}
    for root, _, files in os.walk(poisons_root_path):
        for f in files:
            if f == filename:
                f = os.path.join(root, f)
                if 'target-num-{}/'.format(target_num) in f:
                    # print("reading {}".format(f))
                    with open(f) as resf:
                        res = json.load(resf)
                    poison_label = res['poison_label']
                    target_label = res['target_label']
                    assert len(res['targets'])
                    target_id = list(res['targets'].keys())[0]

                    all_res['poison_label'] = poison_label
                    all_res['target_label'] = target_label
                    all_res['targets'][target_id] = res['targets'][target_id]
    return all_res


def read_attack_stats(poisons_root_path, retrain_epochs, target_num):
    all_res = read_all_in_dir(poisons_root_path, retrain_epochs, target_num)

    if len(all_res['targets']) == 0:
        print("Nothing found in {} when target_num is set to {}".format(poisons_root_path, target_num))
        assert "2000" in poisons_root_path
        print("path {} replaced by".format(poisons_root_path))
        poisons_root_path = poisons_root_path.replace("2000", "1000")
        print(poisons_root_path)
        assert os.path.exists(poisons_root_path)
        all_res = read_all_in_dir(poisons_root_path, retrain_epochs, target_num)

    # assert len(all_res['targets']) == 14
    return all_res


def fetch_poison_bases(poison_label, num_poison, subset, path, transforms):
    """
    Only going to fetch the first num_poison image as the base class from the poison_label class
    """
    img_label_list = torch.load(path)[subset]
    base_tensor_list, base_idx_list = [], []
    for idx, (img, label) in enumerate(img_label_list):
        if label == poison_label:
            base_tensor_list.append(transforms(img))
            base_idx_list.append(idx)
        if len(base_tensor_list) == num_poison:
            break
    return base_tensor_list, base_idx_list