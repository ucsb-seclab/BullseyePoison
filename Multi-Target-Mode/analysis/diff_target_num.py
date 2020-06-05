import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick
from utils import fetch_poison_bases, read_attack_stats
from compare_with_baseline import get_stats
import pandas as pd
from scipy.stats import entropy

LABELSIZE = 14
EPSILON = 0.1
xticks_short = False

TARGETNUMS_LINESTYLES = {1: 'dashed', 5: 'solid', 10: 'dashdot'}
TARGETNUMS_NAMES = {1: r'$N_K=1$', 5: r'$N_K=5$', 10: r'$N_K=10$'}
TARGETNUMS_COLORS = {1: 'crimson', 5: 'darkorange', 10: 'seagreen'}


def diff_target_num(args):
    """
    Simply plot the avg. attack accuracy (over different targets) based on the number of iterations.
    And plot the avg confidence score (over different targets) of the malicious intended class (i.e., poison class)
    based on the number of iterations.
    Same for avg. attack time, avg. clean acc, and avg. loss
    """
    method, root_path, res_path, retrain_epochs, end2end, target_nums = \
        args.method, args.path, args.res_path, args.epochs, args.end2end, args.target_nums
    plot_root_path = "{}/{}/epochs-{}/{}".format(args.res_path, "end2end" if args.end2end else "transfer",
                                                 retrain_epochs, method)

    print("NOTE THAT WE ARE EVALUATING THE CASE THAT THE VICTIMS ARE RETRAINED FOR {} EPOCHS"
          .format(retrain_epochs))
    res = []
    target_ids = None
    for target_num in target_nums:
        r = read_attack_stats(root_path, retrain_epochs=retrain_epochs, target_num=target_num)
        if target_ids is None:
            target_ids = set(r['targets'].keys())
        else:
            target_ids = target_ids.intersection(r['targets'].keys())
        res.append(r)

    target_ids = sorted(list(target_ids))
    print("Target IDs: {}".format(target_ids))

    stats = [get_stats(r, target_ids) for r in res]
    attack_accs = [s[0] for s in stats]
    eval_attack_accs = [s[1] for s in stats]
    clean_acc = [s[2] for s in stats]
    times = [s[3] for s in stats]
    ites = [s[4] for s in stats]
    victims = [s[5] for s in stats]

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
        os.makedirs(plot_root_path)

    for attack_accs1 in attack_accs:
        attack_accs1['meanVictim'] = attack_accs1.mean(axis=1)

    for attack_accs1 in eval_attack_accs:
        attack_accs1['meanVictim'] = attack_accs1.mean(axis=1)

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
            ax.set_ylim([0, MAX_ACC])
        else:
            ax.set_ylim([0, MAX_ACC])

        for attack_accs1, target_num in zip(attack_accs, target_nums):
            ax.plot(ites, attack_accs1[victim], label=TARGETNUMS_NAMES[target_num], color=TARGETNUMS_COLORS[target_num],
                    linewidth=1.7, linestyle=TARGETNUMS_LINESTYLES[target_num])
        tick = mtick.FormatStrFormatter('%d%%')
        ax.yaxis.set_major_formatter(tick)

        if 'mean' in victim:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')

        plt.savefig('{}/{}-attack-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    MAX_ACC = 75
    # plot avg. eval attack acc., one per each victim's network, against unseen different angles of cars
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
            ax.set_ylim([0, 50])
        else:
            ax.set_ylim([0, MAX_ACC])

        for attack_accs1, target_num in zip(eval_attack_accs, target_nums):
            ax.plot(ites, attack_accs1[victim], label=TARGETNUMS_NAMES[target_num], color=TARGETNUMS_COLORS[target_num],
                    linewidth=1.7, linestyle=TARGETNUMS_LINESTYLES[target_num])
        tick = mtick.FormatStrFormatter('%d%%')
        ax.yaxis.set_major_formatter(tick)

        if 'mean' in victim:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE - 2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')

        plt.savefig('{}/{}-eval-attack-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    # plot avg. clean accuracy
    for victim in victims:
        plt.figure(figsize=(8, 4), dpi=400)
        ax = plt.subplot(111)
        ax.set_xlabel('Iterations', fontsize=LABELSIZE)
        ax.set_ylabel('Avg. Clean Test Accuracy - {}'.format(victim), fontsize=LABELSIZE)
        ax.grid(color='black', linestyle='dotted', linewidth=0.4)
        # ax.set_ylim([20, 70])
        for clean_acc1, target_num in zip(clean_acc, target_nums):
            ax.plot(ites, clean_acc1[victim], label=TARGETNUMS_NAMES[target_num], color=TARGETNUMS_COLORS[target_num],
                    linewidth=1.7,
                    linestyle=TARGETNUMS_LINESTYLES[target_num])
        if 'mean' in victim:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')

        plt.savefig('{}/{}-clean-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    # difference from the clean acc before poisoning
    clean_acc_diffs = [(clean_acc1 - clean_acc1.iloc[0]).mean(axis=1) for clean_acc1 in clean_acc]

    plt.figure(figsize=(8, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Avg. Decrease of Clean Test Accuracy', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([20, 70])
    for clean_acc_diff1, target_num in zip(clean_acc_diffs, target_nums):
        ax.plot(ites, clean_acc_diff1, label=TARGETNUMS_NAMES[target_num], color=TARGETNUMS_COLORS[target_num], linewidth=1.7,
                linestyle=TARGETNUMS_LINESTYLES[target_num])
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.savefig('{}/meanVictim-clean-acc-avg-diff.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot avg. time
    plt.figure(figsize=(5, 2.5), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Time (minute)', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([20, 70])
    for times1, target_num in zip(times, target_nums):
        ax.plot(ites, [int(t / 60) for t in times1], label=TARGETNUMS_NAMES[target_num], color=TARGETNUMS_COLORS[target_num],
                linewidth=1.7, linestyle=TARGETNUMS_LINESTYLES[target_num])
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    ax.set_ylim([0, 450])

    plt.savefig('{}/time.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-target Mode Eval')

    parser.add_argument('--method', default='0', choices=["mean", "mean-3Repeat", "mean-5Repeat", "convex"])
    parser.add_argument('--path', default='')
    parser.add_argument('--res-path', default="results/multi/diff-target-nums")
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--target-nums', default=1, type=int, nargs="+", choices=[1, 5, 10])
    parser.add_argument('--end2end', default=0, choices=[0, 1], type=int)

    args = parser.parse_args()

    diff_target_num(args)
