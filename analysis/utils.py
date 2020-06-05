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

COEFFS_COLORS = {'c1': 'black', 'c2': 'crimson', 'c3': 'gray', 'c4': 'blue', 'c5': 'magenta'}
COEFFS_LINESTYLES = {'c1': '-', 'c2': '-.', 'c3': '--', 'c4': '-', 'c5': ':'}
COEFFS_LABELS = {'c1': r'$c_1$ - greatest coefficient', 'c2': r'$c_2$', 'c3': r'$c_3$', 'c4': r'$c_4$',
                 'c5': r'$c_5$ - lowest coefficient'}


def read_attack_stats(poisons_root_path, retrain_epochs):
    filename = 'eval-retrained-for-{}epochs.json'.format(retrain_epochs)
    all_res = {'targets': {}}
    print(poisons_root_path)
    for root, _, files in os.walk(poisons_root_path):
        for f in files:
            if f == filename:
                f = os.path.join(root, f)
                # print("reading {}".format(f))
                with open(f) as resf:
                    res = json.load(resf)
                poison_label = res['poison_label']
                target_label = res['target_label']
                assert len(res['targets'])
                target_id = list(res['targets'].keys())[0]

                all_res['targets'][target_id] = res['targets'][target_id]
    all_res['poison_label'] = poison_label
    all_res['target_label'] = target_label

    # assert len(all_res['targets']) >= 50
    print("stats read for {} different targets".format(len(all_res['targets'])))

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