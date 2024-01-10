import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_FILENAME = 'greedy_scores.p'
FIGURE_FILENAME = 'greedy_results.png'


def figure_plotter() -> None:
    """Plots heatmap of marginal scores for each experiment."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15
    scores = pickle.load(open(DATA_FILENAME, "rb"))

    score_mat = np.zeros((10, 10))
    for i, the_list in enumerate(scores):
        for j, element in enumerate(the_list):
            score_mat[j, i] = "{:.2f}".format(element)

    _, ax = plt.subplots(figsize=(9, 6))
    y_labels = [round(i * 0.1 + 0.1, 1) for i in range(10)]
    ax = sns.heatmap(score_mat, annot=True, linewidths=.5, ax=ax, cmap='gray',
                     cbar_kws={'label': r'$U({\bf d})$'}, yticklabels=y_labels)
    ax.invert_yaxis()
    plt.xlabel('Experiment')
    plt.ylabel(r'${\bf d}$')
    plt.title('Greedy ' + r'$U({\bf d})$' + ' Scores per Experiment')
    plt.savefig(FIGURE_FILENAME)


if __name__ == "__main__":
    figure_plotter()
