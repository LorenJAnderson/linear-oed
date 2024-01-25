from numpy import load
import numpy as np
import tensorflow as tf

events_file = 'logs/DQN_1/events'
evaluations_file = 'logs/evaluations.npz'

train_means = []
for e in tf.compat.v1.train.summary_iterator(events_file):
    for v in e.summary.value:
        if v.tag == 'rollout/ep_rew_mean':
            train_means.append(v.simple_value)
train_means = np.array(train_means[0::50])
print(train_means)

test_means = []
data = load(evaluations_file)
lst = data.files
for item in lst:
    if item == 'results':
        for scores in data[item]:
            test_means.append(np.mean(scores))
test_means = np.mean(np.array(test_means).reshape(-1, 10), axis=1)
print(test_means)
