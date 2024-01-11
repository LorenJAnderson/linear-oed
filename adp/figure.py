import pickle
import matplotlib.pyplot as plt

regressors = pickle.load(open("regressors.p", "rb"))
all_sequences = pickle.load(open("../dp/data.p", "rb"))

buckets = {i: [] for i in range(0, 10)}
for key in all_sequences:
    key_len = len(key)
    prediction = regressors[key_len].predict([key])[0][0]
    actual = all_sequences[key]
    buckets[len(key)].append(actual - prediction)
pickle.dump(buckets, open('errors.p', "wb"))

buckets = pickle.load(open("errors.p", "rb"))
new_buckets = [buckets[i] for i in range(1, 10)]
plt.violinplot(new_buckets, showmeans=True)
plt.show()
