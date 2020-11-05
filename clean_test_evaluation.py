import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import argparse
import os
from models import *
from utils import load_pretrained_net, fetch_target, fetch_nearest_poison_bases, fetch_poison_bases
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


class SubstituteNets:
    def __init__(self, model_resume_path, subs_chk_name, substitute_nets, subs_dp):
        # load the pre-trained models
        self.subs_dp = subs_dp
        self.subs_chk_name = subs_chk_name
        self.substitute_nets = substitute_nets
        self.model_resume_path = model_resume_path
        self._load_models_from_disk()

        print("subs nets, effective num: {}".format(len(self.models)))

    def reload_models(self):
        self._load_models_from_disk()
        print("All {} substitute models are reloaded from the disk!".format(len(self.models)))

    def _load_models_from_disk(self):
        # load the pre-trained models
        sub_net_list = []
        for n_chk, chk_name in enumerate(self.subs_chk_name):
            for snet in self.substitute_nets:
                if args.subs_dp[n_chk] > 0.0:
                    net = load_pretrained_net(snet, chk_name, model_chk_path=self.model_resume_path,
                                              test_dp=self.subs_dp[n_chk])
                elif self.subs_dp[n_chk] == 0.0:
                    net = load_pretrained_net(snet, chk_name, model_chk_path=self.model_resume_path)
                else:
                    assert False
                sub_net_list.append(net)

        self.models = sub_net_list


if __name__ == '__main__':
    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--gpu', default='0', type=str)
    # The substitute models and the victim models
    parser.add_argument('--end2end', default=False, choices=[True, False], type=bool,
                        help="Whether to consider an end-to-end victim")
    parser.add_argument('--substitute-nets', default=['ResNet50', 'ResNet18'], nargs="+", required=False)
    parser.add_argument('--target-net', default=["DenseNet121"], nargs="+", type=str)
    parser.add_argument('--model-resume-path', default='model-chks-release', type=str,
                        help="Path to the pre-trained models")
    parser.add_argument('--net-repeat', default=1, type=int)
    parser.add_argument("--subs-chk-name", default=['ckpt-%s-4800.t7'], nargs="+", type=str)
    parser.add_argument("--test-chk-name", default='ckpt-%s-4800.t7', type=str)
    parser.add_argument('--subs-dp', default=[0], nargs="+", type=float,
                        help='Dropout for the substitute nets, will be turned on for both training and testing')

    # Parameters for poisons
    parser.add_argument('--target-label', default=6, type=int)
    parser.add_argument('--target-index', default=1, type=int,
                        help='index of the target sample')
    parser.add_argument('--poison-label', '-plabel', default=8, type=int,
                        help='label of the poisons, or the target label we want to classify into')
    parser.add_argument('--poison-num', default=5, type=int,
                        help='number of poisons')

    parser.add_argument('--poison-lr', '-plr', default=4e-2, type=float,
                        help='learning rate for making poison')
    parser.add_argument('--poison-momentum', '-pm', default=0.9, type=float,
                        help='momentum for making poison')
    parser.add_argument('--poison-ites', default=4000, type=int,
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
    parser.add_argument('--train-data-path', default='datasets/CIFAR10_TRAIN_Split.pth', type=str,
                        help='path to the official datasets')
    parser.add_argument('--dset-path', default='datasets', type=str,
                        help='path to the official datasets')

    parser.add_argument('--mode', default='convex', type=str,
                        help='if convex, run the convexpolytope attack proposed by the paper, otherwise just run the mean shifting thing')
    parser.add_argument('--retrain-subs-nets', default=False, type=bool,
                        help="It only matters in end2end training mode. If set to True, it means we update the "
                             "substitute models every few steps. Unlike the Convex Polytope attack, We do not apply "
                             "CP loss to multiple layers! ")
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    if args.retrain_subs_nets:
        assert args.end2end

    # Set visible CUDA devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    nets = []
    for n_chk, chk_name in enumerate(args.subs_chk_name):
        for snet in args.substitute_nets:
            net = load_pretrained_net(snet, chk_name, model_chk_path=args.model_resume_path)
            nets.append(net)

    subs_nets = SubstituteNets(args.model_resume_path, args.subs_chk_name, args.substitute_nets, args.subs_dp)

    print("Loading the victims networks")
    targets_net = []
    for tnet in args.target_net:
        target_net = load_pretrained_net(tnet, args.test_chk_name, model_chk_path=args.model_resume_path)
        targets_net.append(target_net)

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    testset = torchvision.datasets.CIFAR10(root=args.dset_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=500)

    for n_chk, chk_name in enumerate(args.subs_chk_name):
        for net_name in args.nets:
            net = load_pretrained_net(snet, chk_name, model_chk_path=args.model_resume_path)
            # Evaluate the results on the clean test set
            val_acc_meter = AverageMeter()
            with torch.no_grad():
                for ite, (input, target) in enumerate(test_loader):
                    input, target = input.to('cuda'), target.to('cuda')

                    output = net(input)

                    prec1 = accuracy(output, target)[0]
                    val_acc_meter.update(prec1.item(), input.size(0))


        print(net_name, chk_name, val_acc_meter.item())
