from utils import read_attack_stats
import pandas as pd


def print_latex(all_res, ite):

    cols = ['epochs', 'sgd', 'dropout', 'retraining_acc', 'poison_acc', 'test_clean_acc', 'attack_acc']
    df = pd.DataFrame(columns=cols)

    for res in all_res:
        poison_label = res['poison_label']

        victims = list(res['targets'][target_ids[0]][ite]['victims'].keys())

        attack_accs_tmp = []
        clean_accs_tmp = []
        retraining_acc_tmp = []
        poison_acc_tmp = []
        for victim in victims:

            vals = [res['targets'][t_id][ite]['victims'][victim]['prediction'] == poison_label for t_id in
                    target_ids]
            attack_accs_tmp.append(100 * (sum(vals) / len(vals)))

            vals = [res['targets'][t_id][ite]['victims'][victim]['clean acc'] for t_id in target_ids]
            clean_accs_tmp.append(sum(vals) / len(vals))

            try:
                vals = [res['targets'][t_id][ite]['victims'][victim]['retraining_acc'] for t_id in target_ids]
                retraining_acc_tmp.append(sum(vals) / len(vals))
            except:
                retraining_acc_tmp.append(-1)
            vals = [res['targets'][t_id][ite]['victims'][victim]['poisons predictions'] for t_id in target_ids]
            vals = [(100.0 * sum([v == poison_label for v in val])) / len(val) for val in vals]
            poison_acc_tmp.append(sum(vals) / len(vals))

        attack_accs_tmp = sum(attack_accs_tmp) / len(victims)
        clean_accs_tmp = sum(clean_accs_tmp) / len(victims)
        retraining_acc_tmp = sum(retraining_acc_tmp) / len(victims)
        poison_acc_tmp = sum(poison_acc_tmp) / len(victims)

        row = [res['epochs'], res['sgd'], res['dropout'],
               retraining_acc_tmp, poison_acc_tmp, clean_accs_tmp, attack_accs_tmp]
        row = {col: r for col, r in zip(cols, row)}
        df = df.append(row, ignore_index=True)

    import IPython
    IPython.embed()


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    method = sys.argv[2]
    assert method in path

    # (epochs, sgd, dropout)
    settings = [(epoch, False, 0.0) for epoch in range(10, 101, 10)] + \
               [(epoch, True, 0.0) for epoch in range(10, 101, 10)] + \
               [(80, False, 0.2), (100, False, 0.2)] + [(180, False, 0.2), (200, False, 0.2)]
    res = []
    target_ids = None
    for epochs, sgd, dropout in settings:
        r = read_attack_stats(path, retrain_epochs=epochs, sgd=sgd, dropout=dropout)
        if target_ids is None:
            target_ids = set(r['targets'].keys())
        else:
            target_ids = target_ids.intersection(r['targets'].keys())
        res.append(r)

    target_ids = sorted(list(target_ids))
    print("Evaluating {}\n Target IDs: {}".format(path, target_ids))

    print_latex(res, ite='4000')
