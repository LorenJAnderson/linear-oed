import pickle

import matplotlib.pyplot as plt
import seaborn as sns


DATA_FILENAME = 'batch_scores.p'
FIGURE_FILENAME = 'batch_results.png'


plt.rcParams["font.size"] = 15
fig = plt.figure(figsize=(6, 8))
design_seq_values = pickle.load(open(DATA_FILENAME, "rb"))
sorted_vals = sorted(design_seq_values.values(), reverse=True)

sns.displot(sorted_vals, color='gray')
plt.xlabel(r'$U({\bf d})$')
plt.ylabel('Count')
plt.title('Distribution of Sorted ' + r'${\bf d}$')
plt.savefig(FIGURE_FILENAME, bbox_inches='tight')