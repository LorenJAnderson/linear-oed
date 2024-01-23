import pickle

import numpy as np

import formulas


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
DES_SEQ_LEN = 10
DATA_FILENAME = 'greedy_scores.p'


def det_best_design(prior_mean: np.array, prior_cov: np.array, obs: tuple,
                    sigma: float) -> tuple:
    """
    Determines design that greedily maximizes marginal expected
    KL-divergence of next experiment.

    Keyword arguments:
    prior_mean -- the prior mean matrix, size 2x2
    prior_cov -- the prior covariance matrix, size 2x2
    obs -- the observation of the experiment
    sig -- the noise factor
    """
    posteriors = [formulas.batch_update(prior_mean, prior_cov, tuple([des]),
                                        obs, sigma)[1] for des in DESIGN_SPACE]
    scores = [formulas.calc_exp_kl(post_cov, prior_cov)
              for post_cov in posteriors]
    best_score, best_des = sorted(zip(scores, DESIGN_SPACE), reverse=True)[0]
    post_mean, post_cov = formulas.batch_update(prior_mean, prior_cov,
                                                tuple([best_des]), obs, sigma)
    return best_des, best_score, post_mean, post_cov, scores


def experiment() -> None:
    """
    Determines greedy design scores for all experiments. Assumes greedy
    actions were taken during all previous experiments when determining
    greedy action for current experiment. Dumps list of scores into
    pickle file.
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
