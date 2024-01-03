import formulas
import numpy as np
from itertools import combinations_with_replacement as cwr
import pickle


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
DES_SEQ_LEN = 10


def init_value_functions():
    all_sequences = []
    for i in range(1, 11):
        all_sequences += list(cwr(DESIGN_SPACE, i))
    return {seq: 0.0 for seq in all_sequences}


def experiment():
    all_sequences = init_value_functions()
    for i in range(10):
        print(i)
        for key in all_sequences.keys():
            if len(key) == 10:
                mean = np.array([[0], [0]])
                cov = np.array([[1, 0], [0, 1]])
                post_mean, post_cov = formulas.batch_update(mean, cov, key,
                                                            tuple([1]*10), 1)
                all_sequences[key] = formulas.calc_exp_kl(post_cov, cov)
            else:
                best_score = -1
                for des in DESIGN_SPACE:
                    unsorted_seq = list(key) + [des]
                    unsorted_seq.sort()
                    new_seq = tuple(unsorted_seq)
                    score = all_sequences[new_seq]
                    if score > best_score:
                        best_score = score
                all_sequences[key] = best_score
    filename = 'data.p'
    pickle.dump(all_sequences, open(filename, "wb"))


if __name__ == "__main__":
    experiment()
