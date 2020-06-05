import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick
from utils import fetch_poison_bases, read_attack_stats, METHODS_COLORS, METHODS_NAMES, METHODS_LINESTYLES
import pandas as pd
from scipy.stats import entropy

LABELSIZE = 14
EPSILON = 0.1
xticks_short = False


def plot_poison_perturbation(paths, methods, plot_root_path, ites, target_ids, poison_label):
    assert len(paths) == len(methods)
    linf = []
    l2 = []
    for path, method in zip(paths, methods):
        linf_tmp, l2_tmp = get_poison_perturbation(path, ites, target_ids, poison_label)
        linf_tmp = [sum(l) / len(linf_tmp[0]) for l in linf_tmp]  # we just plot the MEAN perturbation
        l2_tmp = [sum(l) / len(l2_tmp[0]) for l in l2_tmp]  # we just plot the MEAN perturbation
        linf.append(linf_tmp)
        l2.append(l2_tmp)

    # plot avg. perturbation - linf
    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel(r'Perturbation - $l_{\inf}$', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([20, 70])
    for linf_tmp, method in zip(linf, methods):
        ax.plot(ites, linf_tmp, label=METHODS_NAMES[method], color=METHODS_COLORS[method],
                linewidth=1.7, linestyle=METHODS_LINESTYLES[method])
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.savefig('{}/perturbation-linf.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot avg. perturbation - l2
    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel(r'Perturbation - $l_{2}$', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([20, 70])
    for l21, method1 in zip(l2, methods):
        ax.plot(ites, l21, label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                linewidth=1.7, linestyle=METHODS_LINESTYLES[method1])
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.savefig('{}/perturbation-l2.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()


def get_poison_perturbation(poisons_root_path, ites, target_ids, poison_label,
                            poison_num=5, train_data_path='datasets/CIFAR10_TRAIN_Split.pth', device='cpu'):
    import torch
    import torchvision
    import torchvision.transforms as transforms

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    cifar_mean = torch.Tensor(cifar_mean).reshape((1, 3, 1, 1))
    cifar_std = torch.Tensor(cifar_std).reshape((1, 3, 1, 1))

    base_tensor_list, base_idx_list = fetch_poison_bases(poison_label, poison_num, subset='others',
                                                         path=train_data_path, transforms=transform_test)
    base_tensor_batch = torch.stack(base_tensor_list, 0)
    base_range01_batch = (base_tensor_batch * cifar_std + cifar_mean).view((poison_num, -1))

    linf = []
    l2 = []
    for ite in ites:
        linf_tmp = []
        l2_tmp = []
        for t_id in target_ids:
            path = '{}/{}/poison_{}.pth'.format(poisons_root_path, t_id, "%.5d" % (int(ite) - 1))
            assert os.path.exists(path)
            if device == 'cuda':
                state_dict = torch.load(path)
            elif device == 'cpu':
                state_dict = torch.load(path, map_location=torch.device('cpu'))
            poison_tuple_list, idx_list = state_dict['poison'], state_dict['idx']
            poison_tuple_list = [p for p, _ in poison_tuple_list]
            poison_batch = torch.stack(poison_tuple_list, 0)
            poison_range01_batch = (poison_batch * cifar_std + cifar_mean).view((poison_num, -1))

            for i, ii in zip(idx_list, base_idx_list):
                assert i == ii
            diff = poison_range01_batch - base_range01_batch
            abs_diff = torch.abs(diff)
            linf_diff = torch.max(abs_diff, dim=1).values
            max_perturb = torch.max(linf_diff).item()
            assert max_perturb <= EPSILON + 1e-5, "WHAT THE FUCK, WE HAVE L-inf perturbation of {}".format(max_perturb)

            # linf_tmp.append(torch.mean(linf_diff).item())
            # l2_tmp.append(torch.mean(torch.norm(diff, dim=1)).item())
            linf_tmp.append(sorted(linf_diff.tolist()))
            l2_tmp.append(sorted(torch.norm(diff, dim=1).tolist()))

        linf_sum = []
        l2_sum = []
        for idx in range(poison_num):
            linf_sum.append(sum([l[idx] for l in linf_tmp]) / len(linf_tmp))
            l2_sum.append(sum([l[idx] for l in l2_tmp]) / len(l2_tmp))
        linf.append(linf_sum)
        l2.append(l2_sum)
    return linf, l2


def get_stats(res, target_ids):
    poison_label = res['poison_label']
    ites = sorted([int(ite) for ite in list(res['targets'][target_ids[0]].keys())])
    ites = [str(ite) for ite in ites]
    victims = list(res['targets'][target_ids[0]][ites[0]]['victims'].keys())

    attack_accs = pd.DataFrame(columns=victims)
    attack_accs.index.name = 'ite'
    scores = pd.DataFrame(columns=victims)
    scores.index.name = 'ite'
    clean_acc = pd.DataFrame(columns=victims)
    clean_acc.index.name = 'ite'

    losses = []
    times = []
    coeffs = []
    for ite in ites:
        losses_tmp = [float(res['targets'][t_id][ite]['total_loss']) for t_id in target_ids]
        losses.append(sum(losses_tmp) / len(losses_tmp))
        coeffs.append([res['targets'][t_id][ite]['coeff_list'] for t_id in target_ids])

        # we saved the time performance by mistake fo rehse two targets
        times_tmp = [float(res['targets'][t_id][ite]['time']) for t_id in target_ids if t_id not in ['36', '39']]
        times.append(sum(times_tmp) / len(times_tmp))

        attack_accs_tmp = []
        clean_accs_tmp = []
        scores_tmp = []
        for victim in victims:
            vals = [res['targets'][t_id][ite]['victims'][victim]['scores'][poison_label] for t_id in target_ids]
            scores_tmp.append(sum(vals) / len(vals))

            vals = [res['targets'][t_id][ite]['victims'][victim]['prediction'] == poison_label for t_id in target_ids]
            attack_accs_tmp.append(100 * (sum(vals) / len(vals)))

            vals = [res['targets'][t_id][ite]['victims'][victim]['clean acc'] for t_id in target_ids]
            clean_accs_tmp.append(sum(vals) / len(vals))

        attack_accs.loc[ite] = attack_accs_tmp
        clean_acc.loc[ite] = clean_accs_tmp
        scores.loc[ite] = scores_tmp

    return attack_accs, scores, clean_acc, times, losses, ites, victims, coeffs


def get_entropy(coeffs):
    entropy_list = []
    for coeffs_ite in coeffs:
        ent = .0
        cnt = 0
        for coeffs_target in coeffs_ite:
            for coeffs_net in coeffs_target:
                ent += entropy(coeffs_net, base=2)
                cnt += 1
        entropy_list.append(ent / cnt)
    return entropy_list


def plot_coeffs_entropy(coeffs, methods, ites, plot_root_path):
    entropy_list = [get_entropy(coeffs1) for coeffs1 in coeffs]

    # plot avg. coeff entropy
    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Avg. Entropy of the Coefficients', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([20, 70])

    for entropy1, method1 in zip(entropy_list, methods):
        ax.plot(ites, entropy1, label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                linewidth=1.7, linestyle=METHODS_LINESTYLES[method1])
    ax.legend(loc="lower right", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.xticks(rotation=90)
    plt.savefig('{}/coeffs.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()


def compare_with_baseline(paths, methods, plot_root_path, retrain_epochs):
    """
    Simply plot the avg. attack accuracy (over different targets) based on the number of iterations.
    And plot the avg confidence score (over different targets) of the malicious intended class (i.e., poison class)
    based on the number of iterations.
    Same for avg. attack time, avg. clean acc, and avg. loss
    """

    print("NOTE THAT WE ARE EVALUATING THE CASE THAT THE VICTIMS ARE RETRAINED FOR {} EPOCHS"
          .format(retrain_epochs))
    res = []
    target_ids = None
    for path in paths:
        r = read_attack_stats(path, retrain_epochs=retrain_epochs)
        if target_ids is None:
            target_ids = set(r['targets'].keys())
        else:
            target_ids = target_ids.intersection(r['targets'].keys())
        res.append(r)

    target_ids = sorted(list(target_ids))
    print("Evaluating {}\n Target IDs: {}".format("\n".join(paths), target_ids))

    stats = [get_stats(r, target_ids) for r in res]
    attack_accs = [s[0] for s in stats]
    scores = [s[1] for s in stats]
    clean_acc = [s[2] for s in stats]
    times = [s[3] for s in stats]
    losses = [s[4] for s in stats]
    ites = [s[5] for s in stats]
    victims = [s[6] for s in stats]
    coeffs = [s[7] for s in stats]

    # Just check if we compare same iterations of attacks with each other
    ites_set = set(ites[0])
    for ites_tmp in ites[1:]:
        ites_set = ites_set.union(set(ites_tmp))
    assert len(ites_set) == len(set(ites[0]))
    ites = ites[0]

    # Just check if we compare same victim nets with each other
    victims_set = set(victims[0])
    for victims_tmp in victims[1:]:
        victims_set = victims_set.union(set(victims_tmp))
    assert len(victims_set) == len(set(victims[0]))
    victims = victims[0]

    if not os.path.exists(plot_root_path):
        print("Results folder for saving the plots doesn't exist: {}".format(plot_root_path))
        os.makedirs(plot_root_path)

    for attack_accs1 in attack_accs:
        attack_accs1['meanVictim'] = attack_accs1.mean(axis=1)

    if 'end2end' in paths[0]:
        MAX_ACC = 100
    else:
        MAX_ACC = 75
    # plot avg. attack acc., one per each victim's network
    for counter, victim in enumerate(victims + ['meanVictim']):
        if victim == 'meanVictim':
            plt.figure(figsize=(5, 2.5), dpi=400)
        else:
            plt.figure(figsize=(6, 3), dpi=400)
        ax = plt.subplot(111)
        if victim == 'meanVictim':
            ax.set_xlabel('Iterations', fontsize=LABELSIZE - 2)
            ax.set_ylabel('Attack Success Rate', fontsize=LABELSIZE - 2)
        elif counter == 0:
            ax.set_xlabel('Iterations', fontsize=LABELSIZE)
            ax.set_ylabel('Attack Success Rate', fontsize=LABELSIZE)
        ax.grid(color='black', linestyle='dotted', linewidth=0.4)
        if victim == 'meanVictim':
            ax.set_ylim([0, 60])
        else:
            ax.set_ylim([0, MAX_ACC])

        for attack_accs1, method1 in zip(attack_accs, methods):
            ax.plot(ites, attack_accs1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                    linewidth=1.7, linestyle=METHODS_LINESTYLES[method1])
        tick = mtick.FormatStrFormatter('%d%%')
        ax.yaxis.set_major_formatter(tick)

        if 'mean' in victim or counter == 0:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')
        plt.xticks(rotation=90)
        plt.savefig('{}/{}-attack-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    if methods[0] == 'convex':
        print("Attack Acc. Diff. than CP")
        for attack_accs1, method1 in zip(attack_accs[1:], methods[1:]):
            attack_acc_diff = attack_accs1['meanVictim'] - attack_accs[0]['meanVictim']
            attack_acc_diff_avg = sum(attack_acc_diff) / len(attack_acc_diff)
            print(method1, attack_acc_diff)
            print("attack_acc_diff_avg: ", attack_acc_diff_avg)

    # plot avg. (malicious) class score, one per each victim's network
    for victim in victims:
        plt.figure(figsize=(6, 4), dpi=400)
        ax = plt.subplot(111)
        ax.set_xlabel('Iterations', fontsize=LABELSIZE)
        ax.set_ylabel('Avg. Probability Score of Malicious (i.e., Poison) Class - {}'.format(victim),
                      fontsize=LABELSIZE)
        ax.grid(color='black', linestyle='dotted', linewidth=0.4)
        # ax.set_ylim([20, 70])
        for scores1, method1 in zip(scores, methods):
            ax.plot(ites, scores1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1], linewidth=1.7,
                    linestyle=METHODS_LINESTYLES[method1])
        if 'mean' in victim:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')
        plt.xticks(rotation=90)
        plt.savefig('{}/{}-attack-score-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    # plot avg. clean accuracy
    for victim in victims:
        plt.figure(figsize=(6, 4), dpi=400)
        ax = plt.subplot(111)
        ax.set_xlabel('Iterations', fontsize=LABELSIZE)
        ax.set_ylabel('Avg. Clean Test Accuracy - {}'.format(victim), fontsize=LABELSIZE)
        ax.grid(color='black', linestyle='dotted', linewidth=0.4)
        # ax.set_ylim([20, 70])
        for clean_acc1, method1 in zip(clean_acc, methods):
            ax.plot(ites, clean_acc1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                    linewidth=1.7,
                    linestyle=METHODS_LINESTYLES[method1])
        if 'mean' in victim:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')
        plt.xticks(rotation=90)
        plt.savefig('{}/{}-clean-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    # difference from the clean acc before poisoning
    clean_acc_diffs = [(clean_acc1 - clean_acc1.iloc[0]).mean(axis=1) for clean_acc1 in clean_acc]

    plt.figure(figsize=(5, 2.5), dpi=400)
    ax = plt.subplot(111)
    # ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    # ax.set_ylabel('Decrease in Clean Test Accuracy', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([20, 70])
    for clean_acc_diff1, method1 in zip(clean_acc_diffs, methods):
        ax.plot(ites, clean_acc_diff1, label=METHODS_NAMES[method1], color=METHODS_COLORS[method1], linewidth=1.7,
                linestyle=METHODS_LINESTYLES[method1])
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    plt.xticks(rotation=90)
    plt.savefig('{}/meanVictim-clean-acc-avg-diff.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot avg. time
    plt.figure(figsize=(5, 2.5), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE + 2)
    ax.set_ylabel('Time (minute)', fontsize=LABELSIZE + 2)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([20, 70])
    for times1, method1 in zip(times, methods):
        ax.plot(ites, [int(t / 60) for t in times1], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                linewidth=1.7, linestyle=METHODS_LINESTYLES[method1])
    if 'end2end' not in paths[0]:
        ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.xticks(rotation=90)
    plt.savefig('{}/time.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot coeffs stats
    plot_coeffs_entropy(coeffs, methods, ites, plot_root_path)

    #  plot the amount of the perturbation of poisons
    # plot_poison_perturbation(paths, methods, plot_root_path, ites, target_ids, res[0]['poison_label'])


if __name__ == '__main__':
    import sys

    retrain_epochs = sys.argv[1]
    plots_root_path = sys.argv[2]
    paths = (sys.argv[3:])
    method1 = 'convex'
    method2 = 'mean'
    method3 = 'mean-3'
    method4 = 'mean-5'
    methods = (method1, method2, method3, method4)
    # methods = (method1, method2, method3)
    methods = methods[:len(paths)]
    compare_with_baseline(paths, methods, plots_root_path, retrain_epochs)
