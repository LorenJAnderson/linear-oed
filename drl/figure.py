from numpy import load
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

EVENTS_FILE = 'logs/DQN_1/events'
EVALUATIONS_FILE = 'logs/evaluations.npz'
FIGURE_FILENAME = 'drl_results.png'


def figure_plotter() -> None:
    """Plots line graph of training and testing rewards of DQN agent."""
    train_means = []
    for e in tf.compat.v1.train.summary_iterator(EVENTS_FILE):
        for v in e.summary.value:
            if v.tag == 'rollout/ep_rew_mean':
                train_means.append(v.simple_value)
    train_means = np.array(train_means[0::50])

    test_means = []
    data = load(EVALUATIONS_FILE)
    lst = data.files
    for item in lst:
        if item == 'results':
            for scores in data[item]:
                test_means.append(np.mean(scores))
    test_means = np.mean(np.array(test_means).reshape(-1, 10), axis=1)

    _, ax = plt.subplots(figsize=(6, 4))
    plt.rcParams["font.size"] = 15
    sns.lineplot(train_means, color='gray', linestyle='--', label='Train')
    sns.lineplot(test_means, color='black', label='Test')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.xticks([0, 20, 40, 60, 80, 100], [str(x*40_000) for x in range(6)])
    plt.ylabel('Reward')
    plt.title('DQN Training and Testing Rewards')
    plt.savefig(FIGURE_FILENAME, bbox_inches='tight')


if __name__ == "__main__":
    figure_plotter()
