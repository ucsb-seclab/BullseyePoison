import sys
sys.path.append("../")
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import argparse
import os
from models import *
from utils import load_pretrained_net, fetch_all_external_targets, \
    fetch_nearest_poison_bases, fetch_poison_bases
from trainer import make_convex_polytope_poisons, train_network_with_poison


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


if __name__ == '__main__':
    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--gpu', default='0', type=str)
    # The substitute models and the victim models
    parser.add_argument('--end2end', default=False, choices=[True, False], type=bool,
                        help="Whether to consider an end-to-end victim")
    parser.add_argument('--substitute-nets', default=['ResNet50', 'ResNet18'], nargs="+", required=False)
    parser.add_argument('--victim-net', default=["DenseNet121"], nargs="+", type=str)
    parser.add_argument('--model-resume-path', default='../model-chks-release', type=str,
                        help="Path to the pre-trained models")
    parser.add_argument('--net-repeat', default=1, type=int)
    parser.add_argument("--subs-chk-name", default=['ckpt-%s-4800.t7'], nargs="+", type=str)
    parser.add_argument("--test-chk-name", default='ckpt-%s-4800.t7', type=str)
    parser.add_argument('--subs-dp', default=[0], nargs="+", type=float,
                        help='Dropout for the substitute nets, will be turned on for both training and testing')

    # Parameters for poisons
    parser.add_argument('--target-path', default='../datasets/epfl-gims08/resized/tripod_seq', type=str,
                        help='path to the external images')
    parser.add_argument('--target-index', default=6, type=int,
                        help='model of the car in epfl-gims08 dataset')
    parser.add_argument('--target-start', default='-1', type=int,
                        help='first index of the car in epfl-gims08 dataset')
    parser.add_argument('--target-end', default='-1', type=int,
                        help='last index of the car in epfl-gims08 dataset')
    parser.add_argument('--target-num', default='5', type=int,
                        help='number of targets')

    parser.add_argument('--target-label', default=1, type=int)
    parser.add_argument('--poison-label', '-plabel', default=6, type=int,
                        help='label of the poisons, or the target label we want to classify into')
    parser.add_argument('--poison-num', default=5, type=int,
                        help='number of poisons')

    parser.add_argument('--poison-lr', '-plr', default=4e-2, type=float,
                        help='learning rate for making poison')
    parser.add_argument('--poison-momentum', '-pm', default=0.9, type=float,
                        help='momentum for making poison')
    parser.add_argument('--poison-ites', default=1000, type=int,
                        help='iterations for making poison')
    parser.add_argument('--poison-decay-ites', type=int, metavar='int', nargs="+", default=[])
    parser.add_argument('--poison-decay-ratio', default=0.1, type=float)
    parser.add_argument('--poison-epsilon', '-peps', default=0.1, type=float,
                        help='maximum deviation for each pixel')
    parser.add_argument('--poison-opt', default='adam', type=str)
    parser.add_argument('--nearest', default=False, action='store_true',
                        help="Whether to use the nearest images for crafting the poison")
    parser.add_argument('--subset-group', default=0, type=int)
    parser.add_argument('--original-grad', default=True, choices=[True, False], type=bool)
    parser.add_argument('--tol', default=1e-6, type=float)

    # Parameters for re-training
    parser.add_argument('--retrain-lr', '-rlr', default=0.1, type=float,
                        help='learning rate for retraining the model on poisoned dataset')
    parser.add_argument('--retrain-opt', default='adam', type=str,
                        help='optimizer for retraining the attacked model')
    parser.add_argument('--retrain-momentum', '-rm', default=0.9, type=float,
                        help='momentum for retraining the attacked model')
    parser.add_argument('--lr-decay-epoch', default=[30, 45], nargs="+",
                        help='lr decay epoch for re-training')
    parser.add_argument('--retrain-epochs', default=60, type=int)
    parser.add_argument('--retrain-bsize', default=64, type=int)
    parser.add_argument('--retrain-wd', default=0, type=float)
    parser.add_argument('--num-per-class', default=50, type=int,
                        help='num of samples per class for re-training, or the poison dataset')

    # Checkpoints and resuming
    parser.add_argument('--chk-path', default='chk-black', type=str)
    parser.add_argument('--chk-subdir', default='poisons', type=str)
    parser.add_argument('--eval-poison-path', default='', type=str,
                        help="Path to the poison checkpoint you want to test")
    parser.add_argument('--resume-poison-ite', default=0, type=int,
                        help="Will automatically match the poison checkpoint corresponding to this iteration "
                             "and resume training")
    parser.add_argument('--train-data-path', default='../datasets/CIFAR10_TRAIN_Split.pth', type=str,
                        help='path to the official datasets')
    parser.add_argument('--dset-path', default='datasets', type=str,
                        help='path to the official datasets')

    parser.add_argument('--mode', default='convex', type=str,
                        help='if convex, run the convexpolytope attack proposed by the paper, '
                             'otherwise run the mean method')
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    # Set visible CUDA devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    # load the pre-trained models
    sub_net_list = []
    for n_chk, chk_name in enumerate(args.subs_chk_name):
        for snet in args.substitute_nets:
            if args.subs_dp[n_chk] > 0.0:
                net = load_pretrained_net(snet, chk_name, model_chk_path=args.model_resume_path,
                                          test_dp=args.subs_dp[n_chk])
            elif args.subs_dp[n_chk] == 0.0:
                net = load_pretrained_net(snet, chk_name, model_chk_path=args.model_resume_path)
            else:
                assert False
            sub_net_list.append(net)

    print("subs nets, effective num: {}".format(len(sub_net_list)))

    print("Loading the victims networks")
    targets_net = []
    for vnet in args.victim_net:
        victim_net = load_pretrained_net(vnet, args.test_chk_name, model_chk_path=args.model_resume_path)
        targets_net.append(victim_net)

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # Get the target images
    _targets, targets_indices, eval_targets, val_targets_indices = fetch_all_external_targets(args.target_label,
                                                                                              args.target_path,
                                                                                              args.target_index,
                                                                                              args.target_start,
                                                                                              args.target_end,
                                                                                              args.target_num,
                                                                                              transforms=transform_test)
    targets = [x for x, _ in _targets]
    # assert len(targets) > 1, "we have only one target, but the multiple_target mode is enabled"
    print("All target indices used in crafting poisons: {}".format(targets_indices))

    # creating result directory and log file
    if args.mode == 'mean':
        chk_path = os.path.join(args.chk_path, 'mean')
    else:
        chk_path = os.path.join(args.chk_path, args.mode)
    if args.net_repeat > 1:
        chk_path = '{}-{}Repeat'.format(chk_path, args.net_repeat)
    chk_path = os.path.join(chk_path, str(args.poison_ites))
    chk_path = os.path.join(chk_path, str(args.target_index))
    chk_path = os.path.join(chk_path, "target-num-{}".format(args.target_num))
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    import sys

    sys.stdout = Logger('{}/log.txt'.format(chk_path))

    # Load or craft the poison!
    if args.eval_poison_path != "":
        state_dict = torch.load(args.eval_poison_path)
        poison_tuple_list, base_idx_list = state_dict['poison'], state_dict['idx']
        print("=" * 100)
        print("=" * 100)
        print("Poisons loaded")
        print("Now evaluating on the target nets")
        t = 0
        tt = 0
        assert False, "ToDo: check here!"
    else:
        print(args)
        print("Path: {}".format(chk_path))

        # Otherwise, we craft new poisons
        if args.nearest:
            base_tensor_list, base_idx_list = \
                fetch_nearest_poison_bases(sub_net_list, targets, args.poison_num,
                                           args.poison_label, args.num_per_class, subset='others',
                                           train_data_path=args.train_data_path, transforms=transform_test, )

        else:
            # just fetch the first poison_num samples
            base_tensor_list, base_idx_list = fetch_poison_bases(args.poison_label, args.poison_num, subset='others',
                                                                 path=args.train_data_path, transforms=transform_test)
        base_tensor_list = [bt.to(args.device) for bt in base_tensor_list]
        print("Selected base image indices: {}".format(base_idx_list))
        print("Target indices: {}".format(targets_indices))
        print("# Targets: {}, # Poisons: {}".format(len(targets), len(base_tensor_list)))

        if args.resume_poison_ite > 0:
            state_dict = torch.load(os.path.join(chk_path, "poison_%05d.pth" % args.resume_poison_ite))
            poison_tuple_list, base_idx_list = state_dict['poison'], state_dict['idx']
            poison_init = [pt.to('cuda') for pt, _ in poison_tuple_list]
            # re-direct the results to the resumed dir...
            chk_path += '-resume'
            if not os.path.exists(chk_path):
                os.makedirs(chk_path)
        else:
            poison_init = base_tensor_list

        import time

        t = time.time()
        poison_tuple_list = \
            make_convex_polytope_poisons(sub_net_list, victim_net, base_tensor_list, targets,
                                         targets_indices, device=args.device,
                                         opt_method=args.poison_opt,
                                         lr=args.poison_lr, momentum=args.poison_momentum, iterations=args.poison_ites,
                                         epsilon=args.poison_epsilon, decay_ites=args.poison_decay_ites,
                                         mean=torch.Tensor(cifar_mean).reshape(1, 3, 1, 1),
                                         std=torch.Tensor(cifar_std).reshape(1, 3, 1, 1),
                                         decay_ratio=args.poison_decay_ratio, chk_path=chk_path, end2end=args.end2end,
                                         poison_idxes=base_idx_list, poison_label=args.poison_label, mode=args.mode,
                                         tol=args.tol, start_ite=args.resume_poison_ite, poison_init=poison_init, net_repeat=args.net_repeat)

        tt = time.time()

    res = []
    print("Evaluating against victims networks")
    for vnet, vnet_name in zip(targets_net, args.victim_net):
        print(vnet_name)
        attack_acc, attack_val_acc = train_network_with_poison(vnet, targets, targets_indices, poison_tuple_list,
                                                               base_idx_list, chk_path,
                                                               args, save_state=False, eval_targets=eval_targets)

        res.append((attack_acc, attack_val_acc))
        print("--------")

    print("------SUMMARY------")
    print("TIME ELAPSED (mins): {}".format(int((tt - t) / 60)))
    print("TARGET INDEX: {}".format(args.target_index))
    for tnet_name, (attack_acc, attack_val_acc) in zip(args.victim_net, res):
        print(tnet_name, attack_acc, attack_val_acc)
