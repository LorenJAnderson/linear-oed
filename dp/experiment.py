import pickle
from itertools import combinations_with_replacement as cwr

import numpy as np

import formulas


DES_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
MAX_EXPS = 10
DATA_FILENAME = 'dp_values.p'


def init_value_functions() -> dict:
    """Initializes value function of each design sequence to 0."""
    all_sequences = []
    for exp in range(MAX_EXPS):
        all_sequences += list(cwr(DES_SPACE, exp+1))
    return {seq: 0.0 for seq in all_sequences}


def experiment() -> None:
    """
    Completes 10 dynamic programming updates on each design sequence.
    Terminal states are given value equal to reward of transitioning to
    these states so that these rewards only need to be calculated once.
    Value of start state with length 0 is not computed. Dumps dictionary of
    value functions into pickle file.
    """
    value_dict = init_value_functions()
    for _ in range(MAX_EXPS):
        for key in value_dict.keys():
            if len(key) == MAX_EXPS:
                prior_mean = np.array([[0], [0]])
                prior_cov = np.array([[1, 0], [0, 1]])
                post_mean, post_cov = formulas.batch_update(
                    prior_mean, prior_cov, key, tuple([1]*MAX_EXPS), 1)
                value_dict[key] = formulas.calc_exp_kl(post_cov, prior_cov)
            else:
                next_states = [sorted(list(key) + [des])
                               for des in DES_SPACE]
                all_scores = [value_dict[tuple(state)]
                              for state in next_states]
                value_dict[key] = max(all_scores)
    pickle.dump(value_dict, open(DATA_FILENAME, "wb"))


if __name__ == "__main__":
    experiment()
