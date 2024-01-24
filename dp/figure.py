import pickle

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

DATA_FILENAME = 'dp_values.p'
FIGURE_FILENAME = 'dp_results.png'
MAX_EXPS = 10


def figure_plotter() -> None:
    """Produces ridgeline plot of value function distributions after each
    experiment."""
    value_dict = pickle.load(open(DATA_FILENAME, "rb"))
    exp_values = {exp: [] for exp in range(MAX_EXPS)}
    for exp in range(MAX_EXPS):
        for key in value_dict.keys():
            if len(key) == exp+1:
                exp_values[exp].append(value_dict[key])

    x_axis = np.arange(1.2, 1.8, 0.01)
    plt.rcParams["font.size"] = 15
    fig, axs = plt.subplots(10, figsize=(6, 8))
    fig.suptitle('Progression of Value Distribution')
    fig.supxlabel('Value')
    fig.supylabel('Experiment')
    for exp in range(MAX_EXPS):
        axs[exp].plot(x_axis, gaussian_kde(exp_values[exp])(x_axis))
        axs[exp].set_ylabel(str(exp))
        axs[exp].set_yticks([])
        if exp != MAX_EXPS-1:
            axs[exp].set_xticks([])
    plt.savefig(FIGURE_FILENAME)


if __name__ == "__main__":
    figure_plotter()
