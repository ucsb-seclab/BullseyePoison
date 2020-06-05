import os
import json
import argparse
import numpy as np

import torch.backends.cudnn as cudnn
import torch
import torchvision
import torchvision.transforms as transforms

from utils import load_pretrained_net, fetch_target


class PoisonBatch(torch.nn.Module):
    """
    Implementing this to work with PyTorch optimizers.
    """

    def __init__(self, base_list):
        super(PoisonBatch, self).__init__()
        base_batch = torch.stack(base_list, 0)
        self.poison = torch.nn.Parameter(base_batch.clone())

    def forward(self):
        return self.poison


def proj_onto_simplex(coeffs, psum=1.0):
    """
    Code stolen from https://github.com/hsnamkoong/robustopt/blob/master/src/simple_projections.py
    Project onto probability simplex by default.
    """
    v_np = coeffs.view(-1).detach().cpu().numpy()
    n_features = v_np.shape[0]
    v_sorted = np.sort(v_np)[::-1]
    cssv = np.cumsum(v_sorted) - psum
    ind = np.arange(n_features) + 1
    cond = v_sorted - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w_ = np.maximum(v_np - theta, 0)
    return torch.Tensor(w_.reshape(coeffs.size())).to(coeffs.device)


def least_squares_simplex(A, b, x_init, tol=1e-6, verbose=False, device='cuda'):
    """
    The inner loop of Algorithm 1
    """
    m, n = A.size()
    assert b.size()[0] == A.size()[0], 'Matrix and vector do not have compatible dimensions'

    # Initialize the optimization variables
    if x_init is None:
        x = torch.zeros(n, 1).to(device)
    else:
        x = x_init

    # Define the objective function and its gradient
    f = lambda x: torch.norm(A.mm(x) - b).item()
    # change into a faster version when A is a tall matrix
    AtA = A.t().mm(A)
    Atb = A.t().mm(b)
    grad_f = lambda x: AtA.mm(x) - Atb
    # grad_f = lambda x: A.t().mm(A.mm(x)-b)

    # Estimate the spectral radius of the Matrix A'A
    y = torch.normal(0, torch.ones(n, 1)).to(device)
    lipschitz = torch.norm(A.t().mm(A.mm(y))) / torch.norm(y)

    # The stepsize for the problem should be 2/lipschits.  Our estimator might not be correct, it could be too small.  In
    # this case our learning rate will be too big, and so we need to have a backtracking line search to make sure things converge.
    t = 2 / lipschitz

    # Main iteration
    for iter in range(10000):
        x_hat = x - t * grad_f(x)  # Forward step:  Gradient decent on the objective term
        if f(x_hat) > f(x):  # Check whether the learning rate is small enough to decrease objective
            t = t / 2
        else:
            x_new = proj_onto_simplex(x_hat)  # Backward step: Project onto prob simplex
            stopping_condition = torch.norm(x - x_new) / max(torch.norm(x), 1e-8)
            if verbose: print('iter %d: error = %0.4e' % (iter, stopping_condition))
            if stopping_condition < tol:  # check stopping conditions
                break
            x = x_new
    print(iter)
    return x


def read_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        return {}


if __name__ == '__main__':
    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--gpu', default='0', type=str)
    # The substitute models and the victim models

    parser.add_argument('--model-resume-path', default='model-chks', type=str,
                        help="Path to the pre-trained models")
    parser.add_argument('--substitute-nets', default=['ResNet50', 'ResNet18'], nargs="+", required=False)
    parser.add_argument('--subs-dp', default=[0], nargs="+", type=float,
                        help='Dropout for the substitute nets, will be turned on for both training and testing')
    parser.add_argument("--subs-chk-name", default=['ckpt-%s-4800.t7'], nargs="+", type=str)

    # Parameters for poisons
    parser.add_argument('--target-label', default=6, type=int)
    parser.add_argument('--target-index', default=1, type=int, nargs="+",
                        help='index of the target sample')
    parser.add_argument('--poison-label', '-plabel', default=8, type=int,
                        help='label of the poisons, or the target label we want to classify into')
    parser.add_argument('--poison-num', default=5, type=int,
                        help='number of poisons')

    # Checkpoints and resuming
    parser.add_argument('--eval-poisons-root', default='', type=str,
                        help="Root folder containing poisons crafted for the targets")
    parser.add_argument('--train-data-path', default='datasets/CIFAR10_TRAIN_Split.pth', type=str,
                                    help='path to the official datasets')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--retrain-epochs', default=100, type=int)

    args = parser.parse_args()

    # Set visible CUDA devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        cudnn.benchmark = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # load the pre-trained models
    sub_net_list = []
    for n_chk, chk_name in enumerate(args.subs_chk_name):
        for snet in args.substitute_nets:
            net = load_pretrained_net(snet, chk_name, model_chk_path=args.model_resume_path,
                                      test_dp=args.subs_dp[n_chk], device=args.device)
            sub_net_list.append(net)

    for target_idx in range(0, 80):
        print("computing the coeffs for target {}".format(target_idx))

        target = fetch_target(args.target_label, target_idx, 50, subset='others',
                              path=args.train_data_path, transforms=transform_test)
        json_res_path = '{}/{}/eval-retrained-for-{}epochs.json'.format(args.eval_poisons_root,
                                                                        target_idx, args.retrain_epochs)
        all_res = read_json(json_res_path)
        print(json_res_path)

        assert all_res['poison_label'] == args.poison_label
        assert all_res['target_label'] == args.target_label

        target_feat_list = []
        for net in sub_net_list:
            target_feat_list.append(net(x=target, penu=True).detach())

        for ite, res in all_res['targets'][str(target_idx)].items():
            if len(res['coeff_list']):
                continue
            else:
                poisons_path = '{}/{}/{}'.format(args.eval_poisons_root, target_idx, "poison_%05d.pth" % (int(ite) - 1))
                print("ITE: {}".format(ite))
                print("Loading poisons from {}".format(poisons_path))
                assert os.path.exists(poisons_path)
                if args.device == 'cuda':
                    state_dict = torch.load(poisons_path)
                elif args.device == 'cpu':
                    state_dict = torch.load(poisons_path, map_location=torch.device('cpu'))
                else:
                    assert False
                poison_tuple_list = state_dict['poison']
                poison_init = [pt.to(args.device) for pt, _ in poison_tuple_list]
                n_poisons = len(poison_tuple_list)

                poison_batch = PoisonBatch(poison_init).to(args.device)

                s_coeffs = []
                for net, target_feat in zip(sub_net_list, target_feat_list):
                    s_coeff = torch.ones(n_poisons, 1).to(args.device) / n_poisons

                    pfeat_mat = net(x=poison_batch(), penu=True).t().detach()
                    s_coeffs.append(least_squares_simplex(A=pfeat_mat, b=target_feat.t().detach(),
                                                    x_init=s_coeff, tol=1e-6, device=args.device).view(-1).cpu().detach().tolist())
                assert len(s_coeffs) == len(sub_net_list) == 18
                all_res['targets'][str(target_idx)][ite]['coeff_list'] = s_coeffs

        with open(json_res_path, 'w') as f:
            json.dump(all_res, f)
