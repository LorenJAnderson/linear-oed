import pickle

import numpy as np

import formulas


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
DES_SEQ_LEN = 10
DATA_FILENAME = 'greedy_scores.p'


def det_best_design(mean: np.array, cov: np.array, obs, sig):
    """
    Calculates the expected KL-divergence from the posterior and prior
    covariance matrices.

    Keyword arguments:
    post_cov -- the posterior covariance matrix, size 2x2
    prior_cov -- the prior covariance matrix, size 2x2
    """
    best_score = -1.0
    best_design = 0
    scores = []
    for des in DESIGN_SPACE:
        post_mean, post_cov = formulas.batch_update(mean, cov, [des], obs, sig)
        score = formulas.calc_exp_kl(post_cov, cov)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_design = des
    post_mean, post_cov = formulas.batch_update(mean, cov, [best_design], obs, sig)
    return best_design, best_score, post_mean, post_cov, scores


def experiment() -> None:
    """
    Determines batch design scores for all possible design sequences of given
    length. Dumps dictionary of scores into pickle file.
    """
    greedy_des_seq = []
    all_scores = []
    mean = np.array([[0.0], [0.0]])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    obs = [1.0]*10
    for _ in range(DES_SEQ_LEN):
        design, score, mean, cov, scores = det_best_design(mean, cov, obs, 1.0)
        greedy_des_seq = greedy_des_seq + [design]
        all_scores.append(scores)
    print(greedy_des_seq)
    print(all_scores)
    pickle.dump(all_scores, open(DATA_FILENAME, "wb"))


if __name__ == "__main__":
    experiment()
