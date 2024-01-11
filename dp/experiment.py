import pickle
from itertools import combinations_with_replacement as cwr

import numpy as np

import formulas


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
DES_SEQ_LEN = 10
DATA_FILENAME = 'dp_values.p'


def init_value_functions() -> dict:
    """Initializes value function of each design sequence to 0."""
    all_sequences = []
    for i in range(1, 11):
        all_sequences += list(cwr(DESIGN_SPACE, i))
    return {seq: 0.0 for seq in all_sequences}


def experiment() -> None:
    """Completes 10 dynamic programming updates on each design sequence. Dumps dictionary
    of value functions into pickle file."""
    value_dict = init_value_functions()
    for i in range(10):
        for key in value_dict.keys():
            if len(key) == 10:
                mean = np.array([[0], [0]])
                cov = np.array([[1, 0], [0, 1]])
                post_mean, post_cov = formulas.batch_update(mean, cov, key,
                                                            tuple([1]*10), 1)
                value_dict[key] = formulas.calc_exp_kl(post_cov, cov)
            else:
                next_states = [sorted(list(key) + [des]) for des in DESIGN_SPACE]
                all_scores = [value_dict[tuple(state)] for state in next_states]
                value_dict[key] = max(all_scores)
    pickle.dump(value_dict, open(DATA_FILENAME, "wb"))


if __name__ == "__main__":
    experiment()
