import os
import torch
import scipy.misc
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as transforms


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


def fetch_all_external_targets(target_label, root_path, subset, start_idx, end_idx, num, transforms, device='cpu'):
    import math
    from PIL import Image
    target_list = []
    indices = []
    # target_label = [int(i == target_label) for i in range(10)]
    if start_idx == -1 and end_idx == -1:
        print("No specific indices are determined, so try to include whatever we find")
        idx = 1
        while True:
            path = '{}_{}_{}.jpg'.format(root_path, '%.2d' % subset, '%.3d' % idx)
            if os.path.exists(path):
                img = Image.open(path)
                target_list.append([transforms(img)[None, :, :, :].to(device), torch.tensor([target_label]).to(device)])
                indices.append(idx)
                idx += 1
            else:
                print("In total, we found {} images of target {}".format(len(indices), subset))
                break
    else:
        assert start_idx != -1
        assert end_idx != -1

        for target_index in range(start_idx, end_idx + 1):
            indices.append(target_index)
            path = '{}_{}_{}.jpg'.format(root_path, '%.2d' % subset, '%.3d' % target_index)
            assert os.path.exists(path), "external target couldn't find"
            img = Image.open(path)
            target_list.append([transforms(img)[None, :, :, :].to(device), torch.tensor([target_label]).to(device)])

    i = math.ceil(len(target_list) / num)
    return [t for j, t in enumerate(target_list) if j % i == 0], [t for j, t in enumerate(indices) if j % i == 0], \
           [t for j, t in enumerate(target_list) if j % i != 0], [t for j, t in enumerate(indices) if j % i != 0]


def draw_poisons(eval_poison_path):
    assert os.path.isfile(eval_poison_path), "poisons path doesn't exist"
    state_dict = torch.load(eval_poison_path, map_location=torch.device('cpu'))
    poison_tuple_list, poison_base_idx_list = state_dict['poison'], state_dict['idx']

    mean = torch.Tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1)
    std = torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1)

    os.makedirs('{}/poison_png/'.format(os.path.dirname(eval_poison_path)), exist_ok=True)
    for i, (poison, l) in enumerate(poison_tuple_list):
        poison = poison * std + mean
        save_image(poison, '{}/poison_png/poison-img-{}.png'.format(os.path.dirname(eval_poison_path), i))


def read_poisons(path):
    mean = torch.Tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1)
    std = torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1)
    assert os.path.isfile(path), "poisons path doesn't exist {}".format(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    poison_tuple_list, _ = state_dict['poison'], state_dict['idx']
    poisons = [p * std + mean for p, _ in poison_tuple_list]

    return poisons


def build_row(poisons):
    return torch.cat(poisons, dim=3)


def draw_all(poison_label=6, poison_num=5, target_label=1,
             train_data_path='../datasets/CIFAR10_TRAIN_Split.pth'):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(cifar_mean, cifar_std),
    ])
    bases, _ = fetch_poison_bases(
        poison_label, poison_num, subset='others', path=train_data_path, transforms=transform_test)
    bases = [b.reshape(1, 3, 32, 32) for b in bases]

    imgs = [build_row(bases)]
    for path in ["../chk-black/convex/1000/1/target-num-5/poison_00999.pth",
                 "../chk-black/mean/2000/1/target-num-5/poison_01000.pth",
                 "../chk-black/mean-3Repeat/2000/1/target-num-5/poison_01000.pth",
                 "../chk-black/mean-5Repeat/2000/1/target-num-5/poison_01000.pth"]:
        poisons = read_poisons(path)
        imgs.append(build_row(poisons))
    img = torch.cat(imgs, dim=2)

    white = torch.ones((1, 3, 32 * 5, 32 * 2))
    img = torch.cat([img, white], dim=3)

    imgs = [build_row(bases)]
    for path in ["../chk-black-end2end/convex/1000/1/target-num-5/poison_00999.pth",
                 "../chk-black-end2end/mean/1000/1/target-num-5/poison_00999.pth",
                 "../chk-black-end2end/mean-3Repeat/1000/1/target-num-5/poison_00999.pth"]:
        poisons = read_poisons(path)
        row = build_row(poisons)
        imgs.append(row)
    imgs.append(torch.ones(row.shape))
    rightimg = torch.cat(imgs, dim=2)

    img = torch.cat([img, rightimg], dim=3)

    save_image(img, 'poisons-target-example.png')


def draw_target(target_label=1, target_path="../datasets/epfl-gims08/resized-betterMaybe/tripod_seq",
                 target_index=1, target_start=-1, target_end=-1, target_num=5):
    # cifar_mean = (0.4914, 0.4822, 0.4465)
    # cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(cifar_mean, cifar_std),
    ])

    os.makedirs("target_png", exist_ok=True)

    targets, target_indices, _, _ = fetch_all_external_targets(target_label,
                                                               target_path,
                                                               target_index,
                                                               target_start,
                                                               target_end,
                                                               target_num,
                                                               transforms=transform_test)
    # import IPython
    # IPython.embed()
    for (target, _), ind in zip(targets, target_indices):
        save_image(target, 'target_png/target-{}.png'.format(ind))


if __name__ == "__main__":
    # import sys
    # assert len(sys.argv) > 1
    #
    # eval_poison_path = sys.argv[1]
    # draw_poisons(eval_poison_path)
    #
    draw_target()
    draw_all()
