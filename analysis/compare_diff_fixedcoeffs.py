import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick
from utils import fetch_poison_bases, read_attack_stats
from compare_with_baseline import get_stats
from scipy.stats import entropy

LABELSIZE = 14
EPSILON = 0.1
xticks_short = False


COEFFS = {
    1: [0.4, 0.4, 0.1, 0.1, 0.0],
    2: [0.8, 0.05, 0.05, 0.05, 0.05],
    3: [0.9, 0.025, 0.025, 0.025, 0.025],
    4: [0.6, 0.1, 0.1, 0.1, 0.1],
    5: [0.3, 0.3, 0.3, 0.05, 0.05],
    6: [0.3, 0.2, 0.2, 0.15, 0.15],
    7: [0.22, 0.2, 0.18, 0.17, 0.23],
    8: [0.2, 0.2, 0.2, 0.2, 0.2],
    9: [0.5, 0.3, 0.1, 0.05, 0.05],
    10: [0.5, 0.4, 0.02, 0.02, 0.06],
}


METHODS_LINESTYLES = {1: 'dashed', 2: 'dashed', 3: 'dashed',
                      4: 'dashed', 5: 'dashed', 6: 'dashed',
                      7: 'dashed', 8: 'dashed', 9: 'dashed',
                      10: 'dashed'}
entropies = {t: entropy(coeffs, base=2) for t, coeffs in COEFFS.items()}
l = [(e, c) for c, e in zip(list(COEFFS.keys()), list(entropies.values()))]
l = sorted(l)
METHODS_NAMES = {ty: 'Ent.: %.2f' % ent for index, (ent, ty) in enumerate(l)}
METHODS_COLORS = {1: 'crimson', 2: 'navy', 3: 'green',
                  4: 'black', 5: 'violet', 6: 'red',
                  7: 'orange', 8: 'lightgreen', 9: 'brown',
                  10: 'gray'}

TYPES = [ty for _, ty in l]


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
        os.makedirs(plot_root_path)

    for attack_accs1 in attack_accs:
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
        ax.grid(color='black', linestyle='dotted', linewidth=0.5)
        ax.set_ylim([0, MAX_ACC])

        for attack_accs1, method1 in zip(attack_accs, methods):
            ax.plot(ites, attack_accs1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                    linewidth=1.5, linestyle=METHODS_LINESTYLES[method1])
        tick = mtick.FormatStrFormatter('%d%%')
        ax.yaxis.set_major_formatter(tick)
        if 'mean' in victim or counter == 0:
            ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')

        plt.savefig('{}/{}-attack-acc-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    print("Attack Acc.")
    for attack_accs1, method1 in zip(attack_accs, methods):
        attack_acc = attack_accs1['meanVictim']
        print(method1, attack_acc)

    # plot avg. (malicious) class score, one per each victim's network
    for victim in victims:
        plt.figure(figsize=(8, 4), dpi=400)
        ax = plt.subplot(111)
        ax.set_xlabel('Iterations', fontsize=LABELSIZE)
        ax.set_ylabel('Avg. Probability Score of Malicious (i.e., Poison) Class - {}'.format(victim),
                      fontsize=LABELSIZE)
        ax.grid(color='black', linestyle='dotted', linewidth=0.5)
        # ax.set_ylim([20, 70])
        for scores1, method1 in zip(scores, methods):
            ax.plot(ites, scores1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1], linewidth=1.5,
                    linestyle=METHODS_LINESTYLES[method1])
        ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
        if xticks_short:
            locs, _ = xticks()
            xticks(locs[::5], ites[::5], rotation='vertical')

        plt.savefig('{}/{}-attack-score-avg.pdf'.format(plot_root_path, victim), bbox_inches='tight')
        plt.close()

    # plot avg. clean accuracy
    for victim in victims:
        plt.figure(figsize=(8, 4), dpi=400)
        ax = plt.subplot(111)
        ax.set_xlabel('Iterations', fontsize=LABELSIZE)
        ax.set_ylabel('Avg. Clean Test Accuracy - {}'.format(victim), fontsize=LABELSIZE)
        ax.grid(color='black', linestyle='dotted', linewidth=0.5)
        # ax.set_ylim([20, 70])
        for clean_acc1, method1 in zip(clean_acc, methods):
            ax.plot(ites, clean_acc1[victim], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                    linewidth=1.5,
                    linestyle=METHODS_LINESTYLES[method1])
        ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
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
    ax.set_ylabel('Avg. Clean Test Accuracy Increase/Decrease', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    # ax.set_ylim([20, 70])
    for clean_acc_diff1, method1 in zip(clean_acc_diffs, methods):
        ax.plot(ites, clean_acc_diff1, label=METHODS_NAMES[method1], color=METHODS_COLORS[method1], linewidth=1.5,
                linestyle=METHODS_LINESTYLES[method1])
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.savefig('{}/meanVictim-clean-acc-avg-diff.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot avg. time
    plt.figure(figsize=(8, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Time (minute)', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    # ax.set_ylim([20, 70])
    for times1, method1 in zip(times, methods):
        ax.plot(ites, [int(t / 60) for t in times1], label=METHODS_NAMES[method1], color=METHODS_COLORS[method1],
                linewidth=1.5, linestyle=METHODS_LINESTYLES[method1])
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    plt.savefig('{}/time.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    retrain_epochs = 60
    plots_root_path = 'analysis-results/BP-alternatives-fixedcoeffs/linear-transfer-learning'

    paths = ['attack-results/BP-alternatives-fixedcoeffs/linear-transfer-learning/coeffs_fixed_type_{}'.format(ty) for ty in TYPES]
    methods = [ty for ty in TYPES]

    from plot_polytopes import plot_diff_combinations
    # plot_diff_combinations(TYPES, COEFFS, METHODS_NAMES, '{}/example-figures'.format(plots_root_path))
    compare_with_baseline(paths, methods, plots_root_path, retrain_epochs)