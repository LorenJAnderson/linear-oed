from scipy.stats import gaussian_kde
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

all_sequences = pickle.load(open("data.p", "rb"))
holders = {i: [] for i in range(1, 11)}
for i in range(1, 11):
    for key in all_sequences.keys():
        if len(key) == i:
            holders[i].append(all_sequences[key])
for i in range(1, 11):
    print(len(holders[11-i]))

data = [norm.rvs(loc=i, scale=2, size=50) for i in range(8)]
data = [holders[i] for i in range(1, 11)]

x_axis = np.arange(1.2, 1.8, 0.01)
fig, axs = plt.subplots(10)
fig.suptitle('Ridgeline Plot')
axs[0].plot(x_axis, gaussian_kde(holders[1])(x_axis))
axs[1].plot(x_axis, gaussian_kde(holders[2])(x_axis))
axs[2].plot(x_axis, gaussian_kde(holders[3])(x_axis))
axs[3].plot(x_axis, gaussian_kde(holders[4])(x_axis))
axs[4].plot(x_axis, gaussian_kde(holders[5])(x_axis))
axs[5].plot(x_axis, gaussian_kde(holders[6])(x_axis))
axs[6].plot(x_axis, gaussian_kde(holders[7])(x_axis))
axs[7].plot(x_axis, gaussian_kde(holders[8])(x_axis))
axs[8].plot(x_axis, gaussian_kde(holders[9])(x_axis))
axs[9].plot(x_axis, gaussian_kde(holders[10])(x_axis))
plt.show()