import pickle

import numpy as np

import formulas


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
DES_SEQ_LEN = 10
DATA_FILENAME = 'greedy_scores.p'


def det_best_design(mean: np.array, cov: np.array, obs: tuple, sigma: float) -> tuple:
    """
    Determines design that maximizes marginal KL divergence.

    Keyword arguments:
    post_cov -- the posterior covariance matrix, size 2x2
    prior_cov -- the prior covariance matrix, size 2x2
    obs -- the observation of the experiment
    sig -- the noise factor
    """
    best_score = -1.0
    best_design = 0
    scores = []
    for des in DESIGN_SPACE:
        post_mean, post_cov = formulas.batch_update(mean, cov, tuple([des]), obs, sigma)
        score = formulas.calc_exp_kl(post_cov, cov)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_design = des
    post_mean, post_cov = formulas.batch_update(mean, cov, tuple([best_design]), obs, sigma)
    return best_design, best_score, post_mean, post_cov, scores


def experiment() -> None:
    """
    Determines greedy design scores for all 10 experiments. Assumes greedy actions were
    taken during all previous experiments when determining greedy action for current
    experiment. Dumps dictionary of scores into pickle file.
    """
    greedy_des_seq = []
    all_scores = []
    mean = np.array([[0.0], [0.0]])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    obs = tuple([1.0]*10)
    for _ in range(DES_SEQ_LEN):
        design, score, mean, cov, scores = det_best_design(mean, cov, obs, 1.0)
        greedy_des_seq = greedy_des_seq + [design]
        all_scores.append(scores)
    pickle.dump(all_scores, open(DATA_FILENAME, "wb"))


if __name__ == "__main__":
    experiment()
