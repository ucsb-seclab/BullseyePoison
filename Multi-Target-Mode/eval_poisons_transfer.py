import sys
sys.path.append("../")
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import argparse
import os
from models import *
from dataloader import PoisonedDataset, FeatureSet
from utils import load_pretrained_net, fetch_all_external_targets
import json
import time
import sys
from scipy.special import softmax


def read_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        return {}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_network_with_poison(net, targets, targets_indices, eval_targets, val_targets_indices,
                              poison_tuple_list, poisoned_dset, base_idx_list, args, testset, device='cuda'):
    # requires implementing a get_penultimate_params_list() method to get the parameter identifier of the net's last
    # layer
    if args.end2end:
        params = net.parameters()
    else:
        params = net.module.get_penultimate_params_list()

    if args.retrain_opt == 'adam':
        print("Using Adam for retraining")
        optimizer = torch.optim.Adam(params, lr=args.retrain_lr, weight_decay=args.retrain_wd)
    else:
        print("Using SGD for retraining")
        optimizer = torch.optim.SGD(params, lr=args.retrain_lr, momentum=args.retrain_momentum,
                                    weight_decay=args.retrain_wd)

    net.eval()

    criterion = nn.CrossEntropyLoss().to('cuda')

    poisoned_loader = torch.utils.data.DataLoader(poisoned_dset, batch_size=args.retrain_bsize, shuffle=True)
    # The test set of clean CIFAR10
    test_loader = torch.utils.data.DataLoader(testset, batch_size=500)

    if not args.end2end:
        # create a dataloader that returns the features
        poisoned_loader = torch.utils.data.DataLoader(FeatureSet(poisoned_loader, net, device=args.device),
                                                      batch_size=64, shuffle=True)

    for epoch in range(args.retrain_epochs):
        net.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        time_meter = AverageMeter()

        if epoch in args.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        end_time = time.time()
        for ite, (input, label) in enumerate(poisoned_loader):
            if args.device == 'cuda':
                input, label = input.to('cuda'), label.to('cuda')
            if args.end2end:
                feat = net.module.penultimate(input)
            else:
                feat = input
            output = net.module.linear(feat)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(output, label)[0]

            time_meter.update(time.time() - end_time)
            end_time = time.time()
            loss_meter.update(loss.item(), input.size(0))
            acc_meter.update(prec1.item(), input.size(0))

            if epoch % 50 == 0 and (ite == len(poisoned_loader) - 1):
                print("{2}, Epoch {0}, Iteration {1}, loss {loss.val:.3f} ({loss.avg:.3f}), "
                      "acc {acc.val:.3f} ({acc.avg:.3f})".
                      format(epoch, ite, time.strftime("%Y-%m-%d %H:%M:%S"),
                             loss=loss_meter, acc=acc_meter))
            sys.stdout.flush()

        if epoch == args.retrain_epochs - 1:
            # print the scores for target and base

            print("------------")
            print("Stats after retraining for {} epochs".format(epoch))
            poison_pred_list = []
            for poison_img, _ in poison_tuple_list:
                base_scores = net(poison_img[None, :, :, :].to(device))
                base_score, base_pred = base_scores.topk(1, 1, True, True)
                poison_pred_list.append(base_pred.item())
            print("Target label: {}, Poison label: {}, Poisons' Predictions:{}"
                  .format(args.target_label, args.poison_label, poison_pred_list))
            targets_acc_meter = AverageMeter()
            targets_scores = []
            targets_preds = []
            for idx, target_img in zip(targets_indices, targets):
                # print the scores for target and base
                target_pred = net(target_img.to(device))
                score, pred = target_pred.topk(1, 1, True, True)
                prec = accuracy(target_pred, torch.tensor([args.poison_label]).to(device))[0]
                targets_acc_meter.update(prec.item(), target_img.size(0))
                target_scores = [float(n) for n in list(softmax(target_pred.view(-1).cpu().detach().numpy()))]
                print("Target sample: {}, Prediction:{}, Target's Score:{}"
                      .format(idx, pred[0][0].item(), list(target_scores)))
                targets_scores.append(target_scores)
                targets_preds.append(pred[0][0].item())
            print("Current ATTACK acc on target samples: {acc.avg:.3f}".
                  format(acc=targets_acc_meter))

    # Evaluate the results on the clean test set
    print("-----------------------------------------------")
    print("Now evaluating the network on the clean test set")
    clean_acc_meter = AverageMeter()
    with torch.no_grad():
        for ite, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)

            prec1 = accuracy(output, target)[0]
            clean_acc_meter.update(prec1.item(), input.size(0))

            if ite % 100 == 0 or ite == len(test_loader) - 1:
                print("{2} Epoch {0}, Val iteration {1}, "
                      "acc {acc.val:.3f} ({acc.avg:.3f})".
                      format(epoch, ite, time.strftime("%Y-%m-%d %H:%M:%S"), acc=clean_acc_meter))
    print("* Prec: {}".format(clean_acc_meter.avg))
    print("-----------------------------")

    eval_targets_acc_meter = AverageMeter()
    if eval_targets is not None:
        print("Now evaluating on the extenral target eval set")
        val_targets_preds = []
        val_targets_scores = []
        with torch.no_grad():
            for ite, (input, target) in enumerate(eval_targets):
                input, target = input.to(device), target.to(device)
                output = net(input)
                _, pred = output.topk(1, 1, True, True)
                val_targets_preds.append(str(pred[0][0].item()))
                target_scores = [float(n) for n in list(softmax(output.view(-1).cpu().detach().numpy()))]
                val_targets_scores.append(target_scores)

                prec1 = accuracy(output, torch.tensor([args.poison_label]).to(device))[0]
                eval_targets_acc_meter.update(prec1.item(), input.size(0))
        print("Predictions for the external targets: {}".format(", ".join(val_targets_preds)))
        print("EVAL ATTACK acc: {}".format(eval_targets_acc_meter.avg))
    return {'clean acc': clean_acc_meter.avg, 'targets predictions': targets_preds,
            'poisons predictions': poison_pred_list, 'targets scores': targets_scores,
            'targets malicious scores': [t[args.poison_label] for t in targets_scores],
            'targets attack acc': targets_acc_meter.avg,
            'eval targets attack acc': eval_targets_acc_meter.avg,
            'eval targets predictions': val_targets_preds,
            'eval targets scores': val_targets_scores,
            'eval targets malicious scores': [t[args.poison_label] for t in val_targets_scores],
            'eval targets indices': val_targets_indices,
            'targets indices': targets_indices}


def get_stats(log_path, res, ite):
    with open(log_path) as f:
        log = f.readlines()
    assert "Iteration 0" in log[8]
    date = " ".join(log[8].strip().split()[:2])
    stdate = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    loss = None
    for l in log[8:]:
        if "Iteration {} ".format(ite) in l:
            l = l.strip()
            date = " ".join(l.strip().split()[:2])
            enddate = time.strptime(date, "%Y-%m-%d %H:%M:%S")
            target_loss = l.strip().split()[-1]
            loss = l.strip().split()[6]
            break
    assert loss is not None
    diff = (time.mktime(enddate) - time.mktime(stdate))

    out = {'time': diff}
    out['coeffs_time'] = res.get('coeffs_time', -1)
    out['poisons_time'] = res.get('poisons_time', -1)

    if 'target_loss' in res:
        out['target_loss'] = res['target_loss'].item()
    else:
        out['target_loss'] = target_loss

    if 'total_loss' in res:
        out['total_loss'] = res['total_loss'].item()
    else:
        out['total_loss'] = loss

    if 'coeff_list' in res:
        for net_num, r in enumerate(res['coeff_list']):
            for target_num, rr in enumerate(r):
                if type(rr) == list:
                    res['coeff_list'][net_num][target_num] = [rrr.view(-1).cpu().detach().tolist() for rrr in rr]
                else:
                    res['coeff_list'][net_num][target_num] = rr.view(-1).cpu().detach().tolist()
        out['coeff_list'] = res['coeff_list']
    else:
        out['coeff_list'] = []
    if 'coeff_list_in_victim' in res:
        for net_num, r in enumerate(res['coeff_list_in_victim']):
            for target_num, rr in enumerate(r):
                if type(rr) == list:
                    res['coeff_list_in_victim'][net_num][target_num] = [rrr.view(-1).cpu().detach().tolist() for rrr in rr]
                else:
                    res['coeff_list_in_victim'][net_num][target_num] = rr.view(-1).cpu().detach().tolist()
        out['coeff_list_in_victim'] = res['coeff_list_in_victim']
    else:
        out['coeff_list_in_victim'] = []

    return out


if __name__ == '__main__':
    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--gpu', default='0', type=str)
    # The substitute models and the victim models
    parser.add_argument('--end2end', default=False, choices=[True, False], type=bool,
                        help="Whether to consider an end-to-end victim")
    parser.add_argument('--victim-net', default=["DenseNet121"], nargs="+", type=str)
    parser.add_argument("--test-chk-name", default='ckpt-%s-4800.t7', type=str)
    parser.add_argument('--model-resume-path', default='../model-chks', type=str,
                        help="Path to the pre-trained models")
    parser.add_argument('--subset-group', default=0, type=int)

    # Parameters for poisons
    parser.add_argument('--target-label', default=6, type=int)
    parser.add_argument('--poison-label', '-plabel', default=8, type=int,
                        help='label of the poisons, or the target label we want to classify into')
    parser.add_argument('--poison-num', default=5, type=int,
                        help='number of poisons')
    parser.add_argument('--poison-ites', default=4000, type=int,
                        help='iterations for making poison')
    parser.add_argument('--poison-step', default=50, type=int,
                        help='iterations for making poison')

    parser.add_argument('--target-path', default='../datasets/epfl-gims08/resized/tripod_seq', type=str,
                        help='path to the external images')
    parser.add_argument('--target-subset', default=6, type=int,
                        help='model of the car in epfl-gims08 dataset')
    parser.add_argument('--target-start', default='-1', type=int,
                        help='first index of the car in epfl-gims08 dataset')
    parser.add_argument('--target-end', default='-1', type=int,
                        help='last index of the car in epfl-gims08 dataset')
    parser.add_argument('--target-num', default='5', type=int,
                        help='number of targets')

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
    parser.add_argument('--eval-poisons-root', default='', type=str,
                        help="Root folder containing poisons crafted for the targets")
    parser.add_argument('--train-data-path', default='../datasets/CIFAR10_TRAIN_Split.pth', type=str,
                        help='path to the official datasets')
    parser.add_argument('--dset-path', default='datasets', type=str,
                        help='path to the official datasets')

    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

    # Set visible CUDA devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        cudnn.benchmark = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    testset = torchvision.datasets.CIFAR10(root=args.dset_path, train=False, download=True, transform=transform_test)
    # from torch.utils.data import random_split
    # testset, _ = random_split(testset, (1000, 9000))
    # testset.data = testset.data[:1000] # just for the speed
    target_subset = args.target_subset
    print("Target_subset: {}".format(target_subset))

    # Get the target images
    _targets, targets_indices, eval_targets, val_targets_indices = \
        fetch_all_external_targets(args.target_label,
                                   args.target_path,
                                   args.target_subset,
                                   args.target_start,
                                   args.target_end,
                                   args.target_num,
                                   transforms=transform_test)
    targets = [x for x, _ in _targets]
    # assert len(targets) > 1, "we have only one target, but the multiple_target mode is enabled"
    print("All target indices used in crafting poisons: {}".format(targets_indices))

    json_res_path = '{}/{}/target-num-{}/eval-retrained-for-{}epochs.json'.format(args.eval_poisons_root,
                                                                    target_subset, args.target_num, args.retrain_epochs)

    all_res = read_json(json_res_path)
    # if 'args' in all_res:
    # assert all_res['args'] == str(args)
    all_res['args'] = str(args)
    all_res['poison_label'] = args.poison_label
    all_res['target_label'] = args.target_label
    if 'targets' not in all_res:
        all_res['targets'] = {str(target_subset): {}}

    res = {}
    # ites = list(range(1, args.poison_ites + 1, args.poison_step))
    if args.end2end:
        ites = [1, 51, 101, 201, 301, 401, 601, 801, 1000]
    else:
        if 'convex' in args.eval_poisons_root:
            ites = [1, 51, 101, 201, 301, 401, 601, 801, 1000]
        else:
            ites = [1, 51, 101, 201, 301, 401, 601, 801, 1000]
    # ites = [1, 51, 101, 201, 301, 401, 601, 801, 1201, 1601, 2001, 2401, 3201, 4000]
    # if args.poison_ites not in ites:
    #     ites.append(args.poison_ites)
    no_save = True
    for ite in ites[::-1]:
        if ite in all_res['targets'][str(target_subset)]:
            continue
        poisons_path = '{}/{}/target-num-{}/{}'.format(args.eval_poisons_root, target_subset, args.target_num, "poison_%05d.pth" % (ite - 1))
        print("ITE: {}".format(ite))
        print("Loading poisons from {}".format(poisons_path))
        if not os.path.exists(poisons_path):
            print("skipping target: {}".format(target_subset))
            no_save = False
            break
        if args.device == 'cuda':
            state_dict = torch.load(poisons_path)
        else:
            state_dict = torch.load(poisons_path, map_location=torch.device('cpu'))

        poison_tuple_list, base_idx_list = state_dict['poison'], state_dict['idx']
        print("Poisons loaded")
        poisoned_dset = PoisonedDataset(args.train_data_path, subset='others', transform=transform_train,
                                        num_per_label=args.num_per_class, poison_tuple_list=poison_tuple_list,
                                        poison_indices=base_idx_list, subset_group=args.subset_group)
        print("Poisoned dataset created")
        res[ite] = get_stats('{}/{}/target-num-{}/log.txt'.format(args.eval_poisons_root, target_subset, args.target_num), state_dict, ite - 1)

        res[ite]['victims'] = {}
        for victim_name in args.victim_net:
            print(victim_name)
            victim_net = load_pretrained_net(victim_name, args.test_chk_name, model_chk_path=args.model_resume_path,
                                             device=args.device)
            res[ite]['victims'][victim_name] = \
                train_network_with_poison(victim_net, targets, targets_indices, eval_targets, val_targets_indices,
                                          poison_tuple_list, poisoned_dset, base_idx_list, args, testset)
    if no_save:
        all_res['targets'][target_subset] = res
        all_res['poison_idx_list'] = base_idx_list

        with open(json_res_path, 'w') as f:
            json.dump(all_res, f)
