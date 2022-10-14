import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.weight"] = 'bold'
plt.rcParams["font.size"] = 15

scores = pickle.load(open("data.p", "rb"))
print(scores)
score_mat = np.zeros((10, 10))
for i, the_list in enumerate(scores):
    for j, element in enumerate(the_list):
        score_mat[j, i] = "{:.2f}".format(element)
print(score_mat)
f, ax = plt.subplots(figsize=(9, 6))

yticklabels = [round(i * 0.1 + 0.1, 1) for i in range(10)]
ax = sns.heatmap(score_mat, annot=True, linewidths=.5, ax=ax, cmap='gray', cbar_kws={'label': r'$U({\bf d})$'},
                 yticklabels=yticklabels)
ax.invert_yaxis()
plt.xlabel('Experiment')
plt.ylabel(r'${\bf d}$')
plt.title('Greedy ' + r'$U({\bf d})$' + ' Scores per Experiment')
plt.savefig('greedy.png')