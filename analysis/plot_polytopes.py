import random
import matplotlib.pyplot as plt


def plot_diff_combinations(types, coeffs, methods, plots_path):
    for index, ty in enumerate(types):
        plt.figure(figsize=(4, 4), dpi=600)
        ax = plt.subplot(111)

        c = coeffs[ty]
        random.shuffle(c)
        points = [(0.32, 0.33), (0.32, 0.55), (0.4, 0.65), (0.49, 0.4), (0.47, 0.55)]

        # points.append((sx, sy))
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.scatter(x, y, c='blue', marker='o', s=250)

        sx = 0
        sy = 0
        for (px, py), cc in zip(points, c):
            sx += px * cc
            sy += py * cc

        ax.scatter([sx], [sy], c='crimson', marker='x', s=250)

        plt.xlim((0.31, 0.53))
        plt.ylim((0.31, 0.67))

        for cc, p in zip(c, points):
            ax.annotate(cc, [p[0]-0.01, p[1]+0.02], size=18)

        ax.annotate(methods[ty], [0.4, 0.32], size=18)

        print(ty, c)
        plt.axis('off')
        plt.savefig('{}/{}.pdf'.format(plots_path, index+1), bbox_inches='tight')
        plt.close()
