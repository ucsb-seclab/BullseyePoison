import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick
from utils import read_attack_stats, VICTIMS_COLORS, VICTIMS_LINESTYLES
import pandas as pd

xticks_short = False
LABELSIZE = 14


def plot_attack_avg_stats_over_ites(poisons_root_path, res, target_ids):
    """
    Simply plot the avg. attack accuracy (over different targets) based on the number of iterations.
    And plot the avg confidence score (over different targets) of the malicious intended class (i.e., poison class)
    based on the number of iterations.
    Same for avg. attack time, avg. clean acc, and avg. loss
    """

    poison_label = res['poison_label']

    ites = list(res['targets'][target_ids[0]].keys())
    victims = list(res['targets'][target_ids[0]][ites[0]]['victims'].keys())

    attack_accs = pd.DataFrame(columns=victims)
    attack_accs.index.name = 'ite'
    clean_acc = pd.DataFrame(columns=victims)
    clean_acc.index.name = 'ite'
    eval_attack_accs = pd.DataFrame(columns=victims)
    eval_attack_accs.index.name = 'ite'

    times = []
    for ite in ites:
        times_tmp = [float(res['targets'][t_id][ite]['time']) for t_id in target_ids]
        times.append(sum(times_tmp) / len(times_tmp))

        attack_accs_tmp = []
        clean_accs_tmp = []
        eval_attack_accs_tmp = []
        for victim in victims:

            vals = [res['targets'][t_id][ite]['victims'][victim]['targets attack acc']
                    for t_id in target_ids]
            attack_accs_tmp.append(sum(vals) / len(vals))

            vals = [res['targets'][t_id][ite]['victims'][victim]['eval targets attack acc']
                    for t_id in target_ids]
            eval_attack_accs_tmp.append(sum(vals) / len(vals))

            vals = [res['targets'][t_id][ite]['victims'][victim]['clean acc'] for t_id in target_ids]
            clean_accs_tmp.append(sum(vals) / len(vals))

        attack_accs.loc[ite] = attack_accs_tmp
        clean_acc.loc[ite] = clean_accs_tmp
        eval_attack_accs.loc[ite] = eval_attack_accs_tmp

    plot_root_path = '{}/plots-retrained-for-{}epochs/'.format(poisons_root_path, epochs)
    if not os.path.exists(plot_root_path):
        os.mkdir(plot_root_path)

    # plot avg. attack acc.
    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Avg. Attack Accuracy - Target Set', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    ax.set_ylim([0, 100])
    for victim in victims:
        ax.plot(ites, attack_accs[victim], label=victim, color=VICTIMS_COLORS[victim], linewidth=1.5
                , linestyle=VICTIMS_LINESTYLES[victim])
    tick = mtick.FormatStrFormatter('%d%%')
    ax.yaxis.set_major_formatter(tick)
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    plt.xticks(rotation=90)
    plt.savefig('{}/attack-acc-avg.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel('Iterations', fontsize=LABELSIZE)
    ax.set_ylabel('Avg. Attack Accuracy - Unseen Target Set', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    ax.set_ylim([0, 100])
    for victim in victims:
        ax.plot(ites, eval_attack_accs[victim], label=victim, color=VICTIMS_COLORS[victim], linewidth=1.5
                , linestyle=VICTIMS_LINESTYLES[victim])
    tick = mtick.FormatStrFormatter('%d%%')
    ax.yaxis.set_major_formatter(tick)
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    plt.xticks(rotation=90)
    plt.savefig('{}/eval-attack-acc-avg.pdf'.format(plot_root_path), bbox_inches='tight')
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
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
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
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5, fontsize=9)
    if xticks_short:
        locs, _ = xticks()
        xticks(locs[::5], ites[::5], rotation='vertical')
    print("Avg. time after {}: {}".format(ites[-1], int(times[-1]/60)))
    plt.xticks(rotation=90)
    plt.savefig('{}/time.pdf'.format(plot_root_path), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import sys
    convex_path = sys.argv[1]
    mean_path = sys.argv[2]
    epochs = sys.argv[3]
    assert 'convex' in convex_path
    assert 'mean' in mean_path

    print("NOTE THAT WE ARE EVALUATING THE CASE THAT THE VICTIMS ARE RETRAINED FOR {} EPOCHS"
          .format(epochs))
    convex_res = read_attack_stats(convex_path, retrain_epochs=epochs)
    convex_target_ids = set(convex_res['targets'].keys())
    mean_res = read_attack_stats(mean_path, retrain_epochs=epochs)
    mean_target_ids = set(mean_res['targets'].keys())

    target_ids = convex_target_ids.intersection(mean_target_ids)
    target_ids = sorted(list(target_ids))
    print("Evaluating {} and {}. Target IDs: {}".format(convex_path, mean_path, target_ids))

    plot_attack_avg_stats_over_ites(poisons_root_path=convex_path, res=convex_res, target_ids=target_ids)
    plot_attack_avg_stats_over_ites(poisons_root_path=mean_path, res=mean_res, target_ids=target_ids)

