import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick
from utils import read_attack_stats, VICTIMS_COLORS, VICTIMS_LINESTYLES, \
    COEFFS_COLORS, COEFFS_LINESTYLES, COEFFS_LABELS
import pandas as pd

xticks_short = False
LABELSIZE = 14


def attack_stats_against_camera_targets(poisons_root_path, res, target_ids):

    for print_size in [2, 4]:
        print_size = str(print_size)
        poison_label = res['poison_label']

        ites = list(res['targets'][target_ids[0]].keys())
        victims = list(res['targets'][target_ids[0]][ites[0]]['victims'].keys())

        target_ids = [t_id for t_id in target_ids if
                      len(res['targets'][t_id][ites[0]]['victims'][victims[0]]['camera'])]
        print("Printed size is {}".format(print_size))
        print("target_ids: {}".format(target_ids))

        attack_accs = pd.DataFrame(columns=victims)
        attack_accs.index.name = 'ite'
        scores = pd.DataFrame(columns=victims)
        scores.index.name = 'ite'

        for ite in ites:

            attack_accs_tmp = []
            scores_tmp = []
            for victim in victims:
                vals = [res['targets'][t_id][ite]['victims'][victim]['camera'][print_size]['scores'][poison_label]
                        for t_id in target_ids]
                scores_tmp.append(sum(vals) / len(vals))

                vals = [res['targets'][t_id][ite]['victims'][victim]['camera'][print_size]['prediction'] == poison_label
                        for t_id in target_ids]
                attack_accs_tmp.append(100 * (sum(vals) / len(vals)))

            attack_accs.loc[ite] = attack_accs_tmp
            scores.loc[ite] = scores_tmp

        print("Attack acc. against camera targets for {}".format(poisons_root_path))
        print(attack_accs)
        print("-------------------------")


def get_coeffs_dist(coeffs, ites):
    coeffs_name = ['c1', 'c2', 'c3', 'c4', 'c5']
    coeffs_dist = pd.DataFrame(columns=coeffs_name)
    coeffs_dist.index.name = 'ite'

    for ite, coeffs_ite in zip(ites, coeffs):
        dist = [0] * 5
        for coeffs_target in coeffs_ite:
            for coeffs_net in coeffs_target:
                coeffs_net = sorted(coeffs_net, reverse=True)
                dist = [d + c for d, c in zip(dist, coeffs_net)]
        dist = [d / (len(coeffs_ite) * len(coeffs_ite[0])) for d in dist]

        coeffs_dist.loc[ite] = dist

    return coeffs_name, coeffs_dist


def plot_coeffs_dist(coeffs, ites, plot_root_path):
    coeffs_name, coeffs_dist = get_coeffs_dist(coeffs, ites)

    # plot avg. attack acc.
    plt.figure(figsize=(6, 3), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE-3)
    ax.set_ylabel('Avg. Coefficients Distribution', fontsize=LABELSIZE-3)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    ax.set_ylim([0, 1.0])
    for coeff in coeffs_name:
        ax.plot(ites, coeffs_dist[coeff], label=COEFFS_LABELS[coeff], color=COEFFS_COLORS[coeff],
                linewidth=2, linestyle=COEFFS_LINESTYLES[coeff])
    ax.legend(loc="upper right", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-3)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    plt.xticks(rotation=90)
    plt.savefig('{}/coeffs-dist-avg.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()


def plot_attack_avg_stats_over_ites(poisons_root_path, res, target_ids, print_xylabels=False):
    """
    Simply plot the avg. attack accuracy (over different targets) based on the number of iterations.
    And plot the avg confidence score (over different targets) of the malicious intended class (i.e., poison class)
    based on the number of iterations.
    Same for avg. attack time, avg. clean acc, and avg. loss
    """

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

        # we saved the time performance by mistake fo rehse two targets
        times_tmp = [float(res['targets'][t_id][ite]['time']) for t_id in target_ids if t_id not in ['36', '39']]
        times.append(sum(times_tmp) / len(times_tmp))

        coeffs.append([res['targets'][t_id][ite]['coeff_list'] for t_id in target_ids])

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

    plot_root_path = '{}/plots-retrained-for-{}epochs/'.format(poisons_root_path, epochs)
    if not os.path.exists(plot_root_path):
        os.mkdir(plot_root_path)

    # plot avg. attack acc.
    plt.figure(figsize=(8, 4), dpi=400)
    ax = plt.subplot(111)
    if print_xylabels:
        ax.set_xlabel('Iterations', fontsize=LABELSIZE)
        ax.set_ylabel('Avg. Attack Accuracy', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    ax.set_ylim([0, 80])
    for victim in victims:
        ax.plot(ites, attack_accs[victim], label=victim, color=VICTIMS_COLORS[victim], linewidth=1.5
                , linestyle=VICTIMS_LINESTYLES[victim])
    tick = mtick.FormatStrFormatter('%d%%')
    ax.yaxis.set_major_formatter(tick)
    if print_xylabels:
        ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-1)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    plt.xticks(rotation=90)
    plt.savefig('{}/attack-acc-avg.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot avg. (malicious) class score.
    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Avg. Probability Score of Malicious (i.e., Poison) Class', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    # ax.set_ylim([20, 70])
    for victim in victims:
        ax.plot(ites, scores[victim], label=victim, color=VICTIMS_COLORS[victim], linewidth=1.5,
                linestyle=VICTIMS_LINESTYLES[victim])
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-1)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    plt.xticks(rotation=90)
    plt.savefig('{}/attack-score-avg.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot avg. clean accuracy
    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Avg. Clean Test Accuracy', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    # ax.set_ylim([20, 70])
    for victim in victims:
        ax.plot(ites, clean_acc[victim], label=victim, color=VICTIMS_COLORS[victim], linewidth=1.5,
                linestyle=VICTIMS_LINESTYLES[victim])
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-1)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    plt.xticks(rotation=90)
    plt.savefig('{}/clean-acc-avg.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    # plot avg. time
    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Time (minute)', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    # ax.set_ylim([20, 70])
    ax.plot(ites, [int(t/60) for t in times], label='Time', color='black', linewidth=2)
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=LABELSIZE-1)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')

    print("Avg. time after {}: {}".format(ites[-1], int(times[-1]/60)))
    plt.xticks(rotation=90)
    plt.savefig('{}/time.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    #  plot coeffs dist
    if 'end2end' not in poisons_root_path:
        plot_coeffs_dist(coeffs, ites, plot_root_path)


if __name__ == '__main__':
    import sys
    epochs = sys.argv[1]
    paths = sys.argv[2:]
    assert 'convex' in paths[0]
    assert len(paths) <= 1 or ('mean' in paths[1] and 'mean-' not in paths[1])
    assert len(paths) <= 2 or 'mean-' in paths[2]

    print("NOTE THAT WE ARE EVALUATING THE CASE THAT THE VICTIMS ARE RETRAINED FOR {} EPOCHS"
          .format(epochs))
    res = []
    target_ids = None
    for path in paths:
        r = read_attack_stats(path, retrain_epochs=epochs)
        if target_ids is None:
            target_ids = set(r['targets'].keys())
        else:
            target_ids = target_ids.intersection(r['targets'].keys())
        res.append(r)

    target_ids = sorted(list(target_ids))
    print("Evaluating {}\n Target IDs: {}".format("\n".join(paths), target_ids))

    for path, r in zip(paths, res):
        plot_attack_avg_stats_over_ites(poisons_root_path=path, res=r, target_ids=target_ids,
                                        print_xylabels=True if 'convex' in path else False)

    # print("Now evaluating against camera targets")
    # for path, r in zip(paths, res):
    #     attack_stats_against_camera_targets(poisons_root_path=path, res=r, target_ids=target_ids)