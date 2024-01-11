import pickle

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

DATA_FILENAME = 'dp_values.p'
FIGURE_FILENAME = 'dp_results.png'


def figure_plotter() -> None:
    """Plots ridgeline plot of value function distributions after each experiment."""
    all_sequences = pickle.load(open(DATA_FILENAME, "rb"))
    holders = {i: [] for i in range(1, 11)}
    for i in range(1, 11):
        for key in all_sequences.keys():
            if len(key) == i:
                holders[i].append(all_sequences[key])

    x_axis = np.arange(1.2, 1.8, 0.01)
    plt.rcParams["font.size"] = 15
    fig, axs = plt.subplots(10, figsize=(6, 8))
    fig.suptitle('Progression of Value Distribution')
    fig.supxlabel('Value')
    fig.supylabel('Experiment')
    for i in range(10):
        axs[i].plot(x_axis, gaussian_kde(holders[i+1])(x_axis))
        axs[i].set_ylabel(str(i))
        axs[i].set_yticks([])
        if i != 9:
            axs[i].set_xticks([])
    plt.savefig(FIGURE_FILENAME)


if __name__ == "__main__":
    figure_plotter()
