import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick
from utils import fetch_poison_bases, read_attack_stats, METHODS_COLORS, METHODS_NAMES, METHODS_LINESTYLES
import pandas as pd
from scipy.stats import entropy

LABELSIZE = 14
EPSILON = 0.1
xticks_short = False


def get_stats(res, target_ids):
    poison_label = res['poison_label']
    ites = sorted([int(ite) for ite in list(res['targets'][target_ids[0]].keys())])
    ites = [str(ite) for ite in ites]
    victims = list(res['targets'][target_ids[0]][ites[0]]['victims'].keys())

    attack_accs = pd.DataFrame(columns=victims)
    attack_accs.index.name = 'ite'
    eval_attack_accs = pd.DataFrame(columns=victims)
    eval_attack_accs.index.name = 'ite'
    clean_acc = pd.DataFrame(columns=victims)
    clean_acc.index.name = 'ite'

    times = []
    final_ites = []
    for ite in ites:
        times_tmp = [float(res['targets'][t_id][ite]['time']) for t_id in target_ids]
        times.append(sum(times_tmp) / len(times_tmp))

        eval_attack_accs_tmp = []
        attack_accs_tmp = []
        clean_accs_tmp = []
        for victim in victims:

            vals = [res['targets'][t_id][ite]['victims'][victim]['targets attack acc'] for t_id in target_ids]
            attack_accs_tmp.append((sum(vals) / len(vals)))

            vals = [res['targets'][t_id][ite]['victims'][victim]['eval targets attack acc'] for t_id in target_ids]
            eval_attack_accs_tmp.append((sum(vals) / len(vals)))

            vals = [res['targets'][t_id][ite]['victims'][victim]['clean acc'] for t_id in target_ids]
            clean_accs_tmp.append(sum(vals) / len(vals))

        if ite == '1001':  # just to fix the bug produced while saving the last poisons
            ite = '1000'
        eval_attack_accs.loc[ite] = eval_attack_accs_tmp
        attack_accs.loc[ite] = attack_accs_tmp
        clean_acc.loc[ite] = clean_accs_tmp
        final_ites.append(ite)

        if ite == '1000':
            break

    return attack_accs, eval_attack_accs, clean_acc, times, final_ites, victims


def compare_with_baseline(args):
    """
    Simply plot the avg. attack accuracy (over different targets) based on the number of iterations.
    And plot the avg confidence score (over different targets) of the malicious intended class (i.e., poison class)
    based on the number of iterations.
    Same for avg. attack time, avg. clean acc, and avg. loss
    """
    paths, methods, retrain_epochs = args.paths, args.methods, args.epochs

    plot_root_path = "{}/{}/epochs-{}/target-num-{}".format(args.res_path, "end2end" if args.end2end else "transfer",
                                                            retrain_epochs, args.target_num)

    print("NOTE THAT WE ARE EVALUATING THE CASE THAT THE VICTIMS ARE RETRAINED FOR {} EPOCHS"
          .format(retrain_epochs))
    res = []
    target_ids = None
    for path in paths:
        r = read_attack_stats(path, retrain_epochs=retrain_epochs, target_num=args.target_num)
        if target_ids is None:
            target_ids = set(r['targets'].keys())
        else:
            target_ids = target_ids.intersection(r['targets'].keys())
        res.append(r)

    target_ids = sorted(list(target_ids))
    print("Evaluating {}\n Target IDs: {}".format("\n".join(paths), target_ids))

    stats = [get_stats(r, target_ids) for r in res]
    attack_accs = [s[0] for s in stats]
    eval_attack_accs = [s[1] for s in stats]
    clean_acc = [s[2] for s in stats]
    times = [s[3] for s in stats]
    ites = [s[4] for s in stats]
    victims = [s[5] for s in stats]

    # # 1000 == 1001, saving bug!
    # for ites_tmp in ites:
    #     if sorted(ites_tmp)[-1] == 1001:
    #         for idx in range(len(ites_tmp)):
    #             if ites_tmp[idx] == 1001:
    #                 ites_tmp[idx] = 1000

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

    MAX_ACC = 90
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
            ax.set_ylim([0, 80])
        else:
            ax.set_ylim([0, MAX_ACC])

        for attack_accs1, method1 in zip(attack_accs, methods):
            ax.plot(ites, attack_accs1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                    linewidth=1.7, linestyle=METHODS_LINESTYLES[method1])
        tick = mtick.FormatStrFormatter('%d%%')
        ax.yaxis.set_major_formatter(tick)

        if 'mean' in victim:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')

        plt.savefig('{}/{}-attack-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    if methods[-1] == 'convex':
        print("Attack Acc. Diff. than CP")
        for attack_accs1, method1 in zip(attack_accs[:-1], methods[:-1]):
            attack_acc_diff = attack_accs1['meanVictim'] - attack_accs[-1]['meanVictim']
            attack_acc_diff = sum(attack_acc_diff) / len(attack_acc_diff)
            print(method1, attack_acc_diff)

    MAX_ACC = 60
    # plot avg. attack acc., one per each victim's network, against unseen different angles of cars
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

        for attack_accs1, method1 in zip(eval_attack_accs, methods):
            ax.plot(ites, attack_accs1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                    linewidth=1.7, linestyle=METHODS_LINESTYLES[method1])
        tick = mtick.FormatStrFormatter('%d%%')
        ax.yaxis.set_major_formatter(tick)

        if 'mean' in victim:
            ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE - 2)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')

        plt.savefig('{}/{}-eval-attack-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    print("target-num: {}".format(args.target_num))
    print("EVAL ACC. after {} iterations".format(ites[-1]))
    for attack_accs1, method1 in zip(eval_attack_accs, methods):
        print(method1)
        print(attack_accs1['meanVictim'][-1])
        print("-----")

    if methods[-1] == 'convex':
        print("EVAL Acc. Diff. than CP")
        for attack_accs1, method1 in zip(eval_attack_accs[:-1], methods[:-1]):
            attack_acc_diff = attack_accs1['meanVictim'] - eval_attack_accs[-1]['meanVictim']
            attack_acc_diff = sum(attack_acc_diff) / len(attack_acc_diff)
            print(method1, attack_acc_diff)

    # plot avg. clean accuracy
    for victim in victims:
        plt.figure(figsize=(8, 4), dpi=400)
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
    for clean_acc_diff1, method1 in zip(clean_acc_diffs, methods):
        ax.plot(ites, clean_acc_diff1, label=METHODS_NAMES[method1], color=METHODS_COLORS[method1], linewidth=1.7,
                linestyle=METHODS_LINESTYLES[method1])
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
    for times1, method1 in zip(times, methods):
        ax.plot(ites, [int(t / 60) for t in times1], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                linewidth=1.7, linestyle=METHODS_LINESTYLES[method1])
        print(method1, "time", [int(t / 60) for t in times1][-1])
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-2)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.savefig('{}/time.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-target Mode Eval')

    parser.add_argument('--end2end', default=0, choices=[0, 1], type=int)
    parser.add_argument('--methods', default='0', nargs="+", choices=["mean", "mean-3", "mean-5", "convex"])
    parser.add_argument('--paths', default='', nargs="+")
    parser.add_argument('--res-path', default="../results/multi/diff-methods")
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--target-num', default=1, type=int, choices=[1, 5, 10])
    parser.add_argument('--net-repeats', default=1, type=int, nargs="+", choices=[1, 3, 5])

    args = parser.parse_args()

    assert len(args.paths) == len(args.methods)  #  == len(args.net_repeats)

    compare_with_baseline(args)
