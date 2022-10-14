import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.weight"] = 'bold'
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize=(6, 8))

design_seq_values = pickle.load(open("data.p", "rb"))
sorted_vals = sorted(design_seq_values.values(), reverse=True)
# sns.set(font_scale = 2)
sns.displot(sorted_vals, color='gray')
plt.xlabel(r'$U({\bf d})$')
plt.ylabel('Count')
plt.title('Distribution of Sorted ' + r'${\bf d}$')
plt.savefig('batch.png', bbox_inches='tight')
# plt.show()