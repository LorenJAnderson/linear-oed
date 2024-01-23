import pickle
from itertools import combinations_with_replacement as cwr

import numpy as np

import formulas

DES_SPACE = [0.1 * i for i in range(1, 11)]
DES_SEQ_LEN = 10
ALL_DES_SEQS = list(cwr(DES_SPACE, DES_SEQ_LEN))
DATA_FILENAME = 'batch_scores.p'


def score_sequence(des_seq: tuple) -> float:
    """
    Scores a single batch design sequence with expected KL-divergence.
    Observations are irrelevant to score and are set at default value 1.0.

    Keyword arguments:
    des_seq -- the design sequence of given length
    """
    prior_mean = np.array([[0.0], [0.0]])
    prior_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    obs_seq = tuple([1.0] * DES_SEQ_LEN)
    post_mean, post_cov = formulas.batch_update(prior_mean, prior_cov, des_seq,
                                                obs_seq, 1.0)
    return formulas.calc_exp_kl(post_cov, prior_cov)


def experiment() -> None:
    """
    Determines batch design scores for all possible design sequences of given
    length. Dumps dictionary of scores into pickle file.
    """
    scores = {des_seq: score_sequence(des_seq) for des_seq in ALL_DES_SEQS}
    pickle.dump(scores, open(DATA_FILENAME, "wb"))


if __name__ == "__main__":
    experiment()
