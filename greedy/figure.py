import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_FILENAME = 'greedy_scores.p'
FIGURE_FILENAME = 'greedy_results.png'
TOT_EXPS = 10
DES_SPACE_SIZE = 10


def figure_plotter() -> None:
    """Plots heatmap of marginal scores for each experiment."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15
    scores = pickle.load(open(DATA_FILENAME, "rb"))

    score_mat = np.zeros((TOT_EXPS, DES_SPACE_SIZE))
    for exp, exp_scores in enumerate(scores):
        for des_idx, score in enumerate(exp_scores):
            score_mat[des_idx, exp] = "{:.2f}".format(score)

    _, ax = plt.subplots(figsize=(9, 6))
    y_labels = [round(i * 0.1, 1) for i in range(1, 11)]
    ax = sns.heatmap(score_mat, annot=True, linewidths=.5, ax=ax, cmap='gray',
                     cbar_kws={'label': r'$U({\bf d})$'}, yticklabels=y_labels)
    ax.invert_yaxis()
    plt.xlabel('Experiment')
    plt.ylabel(r'${\bf d}$')
    plt.title('Greedy Marginal ' + r'$U({\bf d})$' + ' per Experiment')
    plt.savefig(FIGURE_FILENAME)


if __name__ == "__main__":
    figure_plotter()
