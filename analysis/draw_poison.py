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


def fetch_target(target_label, target_index, start_idx, path, subset, transforms):
    """
    Fetch the "target_index"-th target, counting starts from start_idx
    """
    img_label_list = torch.load(path)[subset]
    counter = 0
    for idx, (img, label) in enumerate(img_label_list):
        if label == target_label:
            counter += 1
            if counter == (target_index + start_idx + 1):
                if transforms is not None:
                    return transforms(img)[None, :, :, :]
                else:
                    return np.array(img)[None, :, :, :]
    raise Exception("Target with index {} exceeds number of total samples (should be less than {})".format(
        target_index, len(img_label_list) / 10 - start_idx))


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
    assert os.path.isfile(path), "poisons path doesn't exist"
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    poison_tuple_list, _ = state_dict['poison'], state_dict['idx']
    poisons = [p * std + mean for p, _ in poison_tuple_list]

    return poisons


def build_row(poisons):
    return torch.cat(poisons, dim=3)


def draw_all(poison_label=8, poison_num=5, target_index=0, target_label=6,
                                train_data_path='../datasets/CIFAR10_TRAIN_Split.pth'):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(cifar_mean, cifar_std),
    ])
    bases, _ = fetch_poison_bases(
        poison_label, poison_num, subset='others', path=train_data_path, transforms=transform_test)
    bases = [b.reshape(1, 3, 32, 32) for b in bases]

    imgs = [build_row(bases)]
    for path in ["../chk-black/convex/4000/0/poison_03999.pth", "../chk-black-ourmean/mean/4000/0/poison_03999.pth",
                 "../chk-black-ourmean/mean-3Repeat/4000/0/poison_03999.pth",
                 "../chk-black-ourmean/mean-5Repeat/4000/0/poison_03999.pth"]:
        poisons = read_poisons(path)
        imgs.append(build_row(poisons))
    img = torch.cat(imgs, dim=2)

    white = torch.ones((1, 3, 32*5, 32*2))
    img = torch.cat([img, white], dim=3)

    imgs = [build_row(bases)]
    for path in ["../chk-black-end2end/convex/1500/0/poison_01499.pth",
                 "../chk-black-end2end/mean/1500/0/poison_01499.pth",
                 "../chk-black-end2end/mean-3Repeat/1500/0/poison_01499.pth"]:
        poisons = read_poisons(path)
        row = build_row(poisons)
        imgs.append(row)
    imgs.append(torch.ones(row.shape))
    rightimg = torch.cat(imgs, dim=2)

    img = torch.cat([img, rightimg], dim=3)

    save_image(img, 'poisons-target-example.png')


def draw_original_poison_target(poison_label=8, poison_num=5, target_index=0, target_label=6,
                                train_data_path='../datasets/CIFAR10_TRAIN_Split.pth'):
    # cifar_mean = (0.4914, 0.4822, 0.4465)
    # cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(cifar_mean, cifar_std),
    ])
    poison_base_tensor_list, _ = fetch_poison_bases(
        poison_label, poison_num, subset='others', path=train_data_path, transforms=transform_test)

    os.makedirs("original_poison_png", exist_ok=True)
    for i, poison in enumerate(poison_base_tensor_list):
        save_image(poison, 'original_poison_png/poison-base-img-{}.png'.format(i))

    target = fetch_target(target_label, target_index, 50, subset='others',
                          path=train_data_path, transforms=transform_test)

    save_image(target, 'original_poison_png/target-{}.png'.format(target_index))


if __name__ == "__main__":
    # import sys
    # assert len(sys.argv) > 1
    #
    # eval_poison_path = sys.argv[1]
    # draw_poisons(eval_poison_path)
    #
    # draw_original_poison_target()
    draw_all()
