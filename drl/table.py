from numpy import load
import numpy as np
import tensorflow as tf

EVENTS_FILE = 'logs/DQN_1/events'
EVALUATIONS_FILE = 'logs/evaluations.npz'


def table_printer() -> None:
    """Prints training and testing rewards of DQN agent."""
    train_means = []
    for e in tf.compat.v1.train.summary_iterator(EVENTS_FILE):
        for v in e.summary.value:
            if v.tag == 'rollout/ep_rew_mean':
                train_means.append(v.simple_value)
    train_means = np.array(train_means[0::50])
    print(train_means)

    test_means = []
    data = load(EVALUATIONS_FILE)
    lst = data.files
    for item in lst:
        if item == 'results':
            for scores in data[item]:
                test_means.append(np.mean(scores))
    test_means = np.mean(np.array(test_means).reshape(-1, 10), axis=1)
    print(test_means)


if __name__ == "__main__":
    table_printer()
