import pickle

import numpy as np

DATA_FILENAME = 'greedy_scores.p'
TOT_EXPS = 10
DES_SPACE_SIZE = 10


def table_printer() -> None:
    """Prints marginal scores for ten experiments, assuming that the
    previous design sequence was chosen greedily. Rows are individual
    designs from 0.1 to 1.0, and columns are experiments. The
    best design for each experiment is used to form the greedy design
    sequence for subsequent experiments."""
    scores = pickle.load(open(DATA_FILENAME, "rb"))
    score_mat = np.zeros((TOT_EXPS, DES_SPACE_SIZE))
    for exp, exp_scores in enumerate(scores):
        for des_idx, score in enumerate(exp_scores):
            score_mat[des_idx, exp] = "{:.2f}".format(score)
    print(np.flip(score_mat, axis=0))


if __name__ == "__main__":
    table_printer()
