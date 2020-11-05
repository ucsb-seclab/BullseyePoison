import os
import argparse
import pandas as pd
import numpy as np


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import xticks
# import matplotlib.ticker as mtick
#
# from utils import METHODS_COLORS, METHODS_NAMES, METHODS_LINESTYLES
#
#
# LABELSIZE = 14


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


def print_latex_tables(res, ite=800):
    if ite not in res['mean']:
        assert ite == 800
        assert 801 in res['mean']
        ite = 801
    print("Latex table rows for ite: {}".format(ite))

    bp_no_def = res['mean'][ite]['no_def']
    bp_knn_def = res['mean'][ite]['knn_def']
    bp_l2_def = res['mean'][ite]['l2outlier_def']

    cp_no_def = res['convex'][ite]['no_def']
    cp_knn_def = res['convex'][ite]['knn_def']
    cp_l2_def = res['convex'][ite]['l2outlier_def']

    metrics = ['num_deleted_poisons', 'num_deleted_samples', 'adv_succ_rate']

    # First printing rows of knn defense table
    row = [0, '-', '-', '-', '-', np.round(bp_no_def['adv_succ_rate'].item(), 2),
           np.round(cp_no_def['adv_succ_rate'].item(), 2)]
    print(" & ".join([str(r) for r in row]) + " \\\\")

    ks = sorted(bp_knn_def.k.unique())
    # ks = range(1, 21, 2)
    # ks = range(1, 18)
    for k in ks:
        row = ['\\textbf{{{}}}'.format(int(k))]
        for metric in metrics:
            row.append('\\multicolumn{{1}}{{c|}}{{{:.2f}}}'.format(bp_knn_def[bp_knn_def.k == k][metric].item()))
            row.append('{:.2f}'.format(cp_knn_def[cp_knn_def.k == k][metric].item()))
        print(" & ".join([str(r) for r in row]) + " \\\\")

    print("-----End of KNN Defense Table-----")

    # Now printing rows of l2outlier defense table
    row = [0, '-', '-', '-', '-', "{:.2f}".format(bp_no_def['adv_succ_rate'].item()),
           "{:.2f}".format(cp_no_def['adv_succ_rate'].item())]
    print(" & ".join([str(r) for r in row]) + " \\\\")

    fracs = sorted(bp_l2_def.fraction.unique())
    for frac in fracs:
        row = ['\\textbf{{{:.2f}}}'.format(frac)]
        for metric in metrics:
            row.append('\\multicolumn{{1}}{{c|}}{{{:.2f}}}'.format(bp_l2_def[bp_l2_def.fraction == frac][metric].item()))
            row.append('{:.2f}'.format(cp_l2_def[cp_l2_def.fraction == frac][metric].item()))
        print(" & ".join([str(r) for r in row]) + " \\\\")

    print("-----End of l2outlier Defense Table-----")


def parse_results(args):
    res = {}
    for method, res_dir in zip(args.method, args.res_dir):
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
    print_latex_tables(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')
    parser.add_argument('--res-dir', default=[
        'attack-results/100-overlap/linear-transfer-learning/convex/4000/defenses',
        'attack-results/100-overlap/linear-transfer-learning/mean/4000/defenses'], type=str, nargs="+")
    parser.add_argument('--method', default=['convex', 'mean'], type=str, nargs="+")
    parser.add_argument('--plots-dir', default='', type=str)

    args = parser.parse_args()

    parse_results(args)
