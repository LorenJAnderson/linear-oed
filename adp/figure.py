import pickle

import matplotlib.pyplot as plt

TRUE_VALUE_FILENAME = '../dp/dp_values.p'
REGRESSOR_FILENAME = 'regressors.p'
FIGURE_FILENAME = 'adp_results.png'
TOT_EXPS = 10


def figure_plotter() -> None:
    """Plots violin plot of error between estimated and true state-action
    values in the regressors for all design sequences, conditioned on
    experiment number. True values are obtained through dynamic programming."""
    regressors = pickle.load(open("regressors.p", "rb"))
    true_value_dict = pickle.load(open("../dp/dp_values.p", "rb"))
    exp_buckets = {exp: [] for exp in range(TOT_EXPS)}
    for key in true_value_dict:
        current_exp = len(key) - 1
        prediction = regressors[current_exp].predict([key])[0][0]
        actual = true_value_dict[key]
        exp_buckets[current_exp].append(actual - prediction)

    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["font.size"] = 15
    plt.xlabel('Experiment')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error in ADP Regressors')
    plt.violinplot([exp_buckets[exp] for exp in range(TOT_EXPS)],
                   showmeans=True)
    plt.savefig(FIGURE_FILENAME, bbox_inches='tight')


if __name__ == "__main__":
    figure_plotter()
