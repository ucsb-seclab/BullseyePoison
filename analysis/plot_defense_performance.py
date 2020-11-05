import os
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick

LABELSIZE = 14

# def plot(res, ite, plots_dir):
#
#     if not os.path.exists(plots_dir):
#         os.makedirs(plots_dir)
#
#     print("Plots for ite: {}".format(ite))
#
#     # knn defense, different k
#     plt.figure(figsize=(6, 4), dpi=400)
#     ax = plt.subplot(111)
#     ax.set_xlabel('Iterations', fontsize=LABELSIZE)
#     ax.set_ylabel(r'Perturbation - $l_{\inf}$', fontsize=LABELSIZE)
#     ax.grid(color='black', linestyle='dotted', linewidth=0.4)
#     # ax.set_ylim([20, 70])
#     for method, method_res in res.items():
#         method_no_def = method_res['no_def']
#         ax.plot(ites, linf_tmp, label=METHODS_NAMES[method], color=METHODS_COLORS[method],
#                 linewidth=1.7, linestyle=METHODS_LINESTYLES[method])
#     ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE - 2)


COLORS = {5: 'black', 10: 'darkcyan', 25: 'crimson'}


def plot(res, num_poisons, plots_dir, ITE=800):
    # import IPython
    # IPython.embed()
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # print("Plots for ite: {}".format(ite))
    methods = res.keys()

    # KNN DEFENSE
    # ks = list(range(1, 62, 2)) + list(range(61, 106, 6))
    ks = list(range(1, 102, 2)) + list(range(121, 202, 20))

    #  attack success rate
    plt.figure(figsize=(8, 3), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel(r'$k$', fontsize=LABELSIZE)
    ax.set_ylabel('Attack Success (%)', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)

    for method, num_poison in zip(methods, num_poisons):
        # ite = max(res[method].keys())
        if num_poison == 5:
            assert ITE == 800
            ite = 801
        else:
            ite = ITE
        method_res = res[method][ite]
        df = method_res['knn_def']
        data = [method_res['no_def']['adv_succ_rate'].item()] + \
               [df[df.k == k]['adv_succ_rate'].item() for k in ks]
        x = [0] + ks
        ax.plot(x, data, label="{} Poisons".format(num_poison), color=COLORS[num_poison],
                linewidth=1.7)
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE - 2)
    tick = mtick.FormatStrFormatter('%d%%')
    ax.yaxis.set_major_formatter(tick)
    tick = mtick.FormatStrFormatter('%d')
    ax.xaxis.set_major_formatter(tick)
    ax.set_ylim([0, 100])

    plt.savefig('{}/knn-attack-succ-rate.pdf'.format(plots_dir), bbox_inches='tight')
    plt.close()

    #  Poison Detection Precision and Recall
    plt.figure(figsize=(8, 3), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel(r'$k$', fontsize=LABELSIZE)
    ax.set_ylabel('Poison Detection Rates (%)', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)
    # ax.set_ylim([0, 90])

    for method, num_poison in zip(methods, num_poisons):
        if num_poison == 5:
            assert ITE == 800
            ite = 801
        else:
            ite = ITE
        method_res = res[method][ite]
        df = method_res['knn_def']
        prec = [0] + [100.0 * (df[df.k == k]['num_deleted_poisons'].item()
                               / df[df.k == k]['num_deleted_samples'].item())
                      for k in ks]
        x = [0] + ks
        ax.plot(x, prec, label="{} Poisons - Precision".format(num_poison), color=COLORS[num_poison],
                linewidth=1.7)

        recall = [0] + [100.0 * (df[df.k == k]['num_deleted_poisons'].item() / num_poison)
                        for k in ks]
        x = [0] + ks
        ax.plot(x, recall, label="{} Poisons - Recall".format(num_poison), color=COLORS[num_poison],
                linewidth=1.7, linestyle='--')

    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE - 2)
    tick = mtick.FormatStrFormatter('%d%%')
    ax.yaxis.set_major_formatter(tick)
    tick = mtick.FormatStrFormatter('%d')
    ax.xaxis.set_major_formatter(tick)

    plt.savefig('{}/knn-poison-detection-rates.pdf'.format(plots_dir), bbox_inches='tight')
    plt.close()

    # END OF KNN DEFENSE

    # L2 Outlier DEFENSE
    fracs = list(np.arange(0.02, 0.41, 0.02))

    #  attack success rate
    plt.figure(figsize=(8, 3), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel(r'$\mu$', fontsize=LABELSIZE)
    # ax.set_ylabel('Attack Success (%)', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)

    for method, num_poison in zip(methods, num_poisons):
        if num_poison == 5:
            assert ITE == 800
            ite = 801
        else:
            ite = ITE
        method_res = res[method][ite]
        df = method_res['l2outlier_def']
        data = [method_res['no_def']['adv_succ_rate'].item()] + \
               [df[df.fraction == frac]['adv_succ_rate'].item() for frac in fracs]
        x = [0] + fracs
        ax.plot(x, data, label="{} Poisons".format(num_poison), color=COLORS[num_poison],
                linewidth=1.7)
    # ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE - 2)
    tick = mtick.FormatStrFormatter('%d%%')
    ax.yaxis.set_major_formatter(tick)
    # tick = mtick.FormatStrFormatter('%d')
    # ax.xaxis.set_major_formatter(tick)

    plt.savefig('{}/l2-attack-succ-rate.pdf'.format(plots_dir), bbox_inches='tight')
    plt.close()

    #  Poison Detection Precision and Recall
    plt.figure(figsize=(8, 3), dpi=400)
    ax = plt.subplot(111)
    ax.set_xlabel(r'$\mu$', fontsize=LABELSIZE)
    # ax.set_ylabel('Poison Detection Precision and Recall (%)', fontsize=LABELSIZE)
    ax.grid(color='black', linestyle='dotted', linewidth=0.4)

    for method, num_poison in zip(methods, num_poisons):
        if num_poison == 5:
            assert ITE == 800
            ite = 801
        else:
            ite = ITE
        method_res = res[method][ite]
        df = method_res['l2outlier_def']
        prec = [0] + [100.0 * (df[df.fraction == frac]['num_deleted_poisons'].item()
                               / df[df.fraction == frac]['num_deleted_samples'].item())
                      for frac in fracs]
        x = [0] + fracs
        ax.plot(x, prec, label="{} Poisons - Precision".format(num_poison), color=COLORS[num_poison],
                linewidth=1.7)

        recall = [0] + [100.0 * (df[df.fraction == frac]['num_deleted_poisons'].item() / num_poison)
                        for frac in fracs]
        x = [0] + fracs
        ax.plot(x, recall, label="{} Poisons - Recall".format(num_poison), color=COLORS[num_poison],
                linewidth=1.7, linestyle='--')

    # ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=LABELSIZE - 2)
    tick = mtick.FormatStrFormatter('%d%%')
    ax.yaxis.set_major_formatter(tick)
    # tick = mtick.FormatStrFormatter('%d')
    # ax.xaxis.set_major_formatter(tick)

    plt.savefig('{}/l2-poison-detection-rates.pdf'.format(plots_dir), bbox_inches='tight')
    plt.close()

    # END OF L2 Outlier DEFENSE


def parse_results(args):
    res = {}
    for num_poison, res_dir in zip(args.num_poisons, args.res_dir):
        method = 'mean-{}poisons'.format(num_poison)
        no_defense_res_path = "{}/no-defense.pickle".format(res_dir)
        assert os.path.exists(no_defense_res_path)
        no_defense_res = pd.read_pickle(no_defense_res_path)

        knn_defense_res_path = "{}/knn.pickle".format(res_dir)
        assert os.path.exists(knn_defense_res_path)
        knn_defense_res = pd.read_pickle(knn_defense_res_path)
        knn_defense_res['num_deleted_poisons'] = knn_defense_res.apply(
            lambda row: len(row.deleted_poisons_base_indices),
            axis=1)

        l2outlier_defense_res_path = "{}/l2outlier.pickle".format(res_dir)
        assert os.path.exists(l2outlier_defense_res_path)
        l2outlier_defense_res = pd.read_pickle(l2outlier_defense_res_path)
        l2outlier_defense_res['num_deleted_poisons'] = l2outlier_defense_res.apply(
            lambda row: len(row.deleted_poisons_base_indices),
            axis=1)

        ites = no_defense_res.ite.unique()
        victims = no_defense_res.victim_net.unique()

        res[method] = {}

        for ite in ites:
            print("Evaluating the results for poison-ite: {}".format(ite))

            nodef_cols = ['ite', 'adv_succ_rate', 'clean_test_acc']
            no_def_avg_res = pd.DataFrame(columns=nodef_cols)

            knn_cols = ['ite', 'k', 'adv_succ_rate', 'clean_test_acc',
                        'num_deleted_samples', 'num_deleted_poisons']
            knn_def_avg_res = pd.DataFrame(columns=knn_cols)

            l2_cols = ['ite', 'fraction', 'adv_succ_rate', 'clean_test_acc',
                       'num_deleted_samples', 'num_deleted_poisons']
            l2outlier_def_avg_res = pd.DataFrame(columns=l2_cols)

            no_def = no_defense_res[no_defense_res.ite == ite]
            knn_def = knn_defense_res[knn_defense_res.ite == ite]
            l2_def = l2outlier_defense_res[l2outlier_defense_res.ite == ite]

            assert len(no_def.victim_net.unique()) == len(victims) == 8

            print("When there is no defense employed!")
            victims_net_adv_succ_mean = [no_def[no_def.victim_net == victim].victim_net_adv_succ.mean() for victim in
                                         victims]
            adv_succ_mean_acc = 100.0 * (sum(victims_net_adv_succ_mean) / len(victims))
            print("Adversarial Success Rate - Mean: {:.4f}".format(adv_succ_mean_acc))

            victims_net_test_acc_mean = [no_def[no_def.victim_net == victim].victim_net_test_acc.mean() for victim in
                                         victims]
            test_acc_mean = sum(victims_net_test_acc_mean) / len(victims)
            print("Clean Test Accuracy - Mean: {:.4f}".format(test_acc_mean))

            row = [ite, adv_succ_mean_acc, test_acc_mean]
            row = {col: v for col, v in zip(nodef_cols, row)}
            no_def_avg_res = no_def_avg_res.append(row, ignore_index=True)
            print("+" * 30)

            ks = knn_def.k.unique()
            for k in ks:
                d = knn_def[knn_def.k == k]
                assert len(d.victim_net.unique()) == len(victims) == 8

                print("In knn defense, when k is set to: {}".format(k))
                victims_net_adv_succ_mean = [d[d.victim_net == victim].victim_net_adv_succ.mean() for victim in
                                             victims]
                adv_succ_mean_acc = 100.0 * (sum(victims_net_adv_succ_mean) / len(victims))
                print("Adversarial Success Rate - Mean: {:.4f}".format(adv_succ_mean_acc))

                victims_net_test_acc_mean = [d[d.victim_net == victim].victim_net_test_acc.mean() for victim in
                                             victims]
                test_acc_mean = sum(victims_net_test_acc_mean) / len(victims)
                print("Clean Test Accuracy - Mean: {:.4f}".format(test_acc_mean))

                victims_num_deleted_samples_mean = [d[d.victim_net == victim].num_deleted_samples.mean() for victim in
                                                    victims]
                num_deleted_samples_mean = sum(victims_num_deleted_samples_mean) / len(victims)
                print("In total, {:.4f} samples are deleted!".format(num_deleted_samples_mean))

                victims_num_deleted_poisons_mean = [d[d.victim_net == victim].num_deleted_poisons.mean() for victim in
                                                    victims]
                num_deleted_poisons_mean = sum(victims_num_deleted_poisons_mean) / len(victims)
                print("In total, {:.4f} poisons are deleted!".format(num_deleted_poisons_mean))

                row = [ite, k, adv_succ_mean_acc, test_acc_mean, num_deleted_samples_mean, num_deleted_poisons_mean]
                row = {col: v for col, v in zip(knn_cols, row)}
                knn_def_avg_res = knn_def_avg_res.append(row, ignore_index=True)

            print("+" * 30)

            fracs = l2_def.fraction.unique()
            for frac in fracs:
                d = l2_def[l2_def.fraction == frac]
                assert len(d.victim_net.unique()) == len(victims) == 8

                print("In l2outlier defense, when fraction is set to: {:.4f}".format(frac))
                victims_net_adv_succ_mean = [d[d.victim_net == victim].victim_net_adv_succ.mean() for victim in
                                             victims]
                adv_succ_mean_acc = 100.0 * (sum(victims_net_adv_succ_mean) / len(victims))
                print("Adversarial Success Rate - Mean: {:.4f}".format(adv_succ_mean_acc))

                victims_net_test_acc_mean = [d[d.victim_net == victim].victim_net_test_acc.mean() for victim in
                                             victims]
                test_acc_mean = sum(victims_net_test_acc_mean) / len(victims)
                print("Clean Test Accuracy - Mean: {:.4f}".format(test_acc_mean))

                victims_num_deleted_samples_mean = [d[d.victim_net == victim].num_deleted_samples.mean() for victim in
                                                    victims]
                num_deleted_samples_mean = sum(victims_num_deleted_samples_mean) / len(victims)
                print("In total, {:.4f} samples are deleted!".format(num_deleted_samples_mean))

                victims_num_deleted_poisons_mean = [d[d.victim_net == victim].num_deleted_poisons.mean() for victim in
                                                    victims]
                num_deleted_poisons_mean = sum(victims_num_deleted_poisons_mean) / len(victims)
                print("In total, {:.4f} poisons are deleted!".format(num_deleted_poisons_mean))

                row = [ite, frac, adv_succ_mean_acc, test_acc_mean, num_deleted_samples_mean, num_deleted_poisons_mean]
                row = {col: v for col, v in zip(l2_cols, row)}
                l2outlier_def_avg_res = l2outlier_def_avg_res.append(row, ignore_index=True)

            print("+" * 30)

            res[method][ite] = {'no_def': no_def_avg_res, 'knn_def': knn_def_avg_res,
                                'l2outlier_def': l2outlier_def_avg_res}
        print("-" * 50)

    # import IPython
    # IPython.embed()
    # plot(res, ites[-1])
    plot(res, args.num_poisons, args.plots_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--res-dir', default=[
        'attack-results/100-overlap/linear-transfer-learning/mean/4000/defenses',
        'attack-results-10poisons/100-overlap/linear-transfer-learning/mean/800/defenses',
        'attack-results-25poisons/100-overlap/linear-transfer-learning/mean/800/defenses'], type=str, nargs="+")
    parser.add_argument('--num-poisons', default=[5, 10, 25], type=int, nargs="+")
    parser.add_argument('--plots-dir', default='analysis-results/defenses/BP', type=str)

    args = parser.parse_args()

    # args.res_dir = args.res_dir[1:]
    # args.num_poisons = args.num_poisons[1:]

    parse_results(args)
