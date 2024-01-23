import numpy as np
import scipy.linalg as lin


def calc_exp_kl(post_cov: np.array, prior_cov: np.array) -> float:
    """
    Calculates the expected KL-divergence between the posterior and prior
    covariance matrices. Note that expected KL-divergence is equivalent to
    KL-divergence in the linear regression case because the KL-divergence is
    independent of the observations and entirely dependent on the designs.

    Keyword arguments:
    post_cov -- the posterior covariance matrix, size 2x2
    prior_cov -- the prior covariance matrix, size 2x2
    """
    return 0.5 * (np.log(lin.det(prior_cov)) - np.log(lin.det(post_cov)))


def bayesian_update(prior_mean: np.array, prior_cov: np.array,
                    design_mat: np.array, obs: float, sigma: float) -> tuple:
    """
    Calculates the posterior mean vector and posterior covariance matrix
    after a single experiment.

    Keyword arguments:
    prior_mean -- the prior mean matrix, size 2x2
    prior_cov -- the prior covariance matrix, size 2x2
    design_mat -- the design matrix, size 2x1
    obs -- the observation of the experiment
    sigma -- the noise factor
    """
    a_inv = lin.inv(lin.inv(prior_cov) + (
            sigma ** (-2.0) * np.dot(design_mat, np.transpose(design_mat))))
    b = (np.dot(lin.inv(prior_cov), prior_mean) +
         (sigma ** (-2.0) * design_mat * obs))
    return np.dot(a_inv, b), a_inv


def batch_update(mean: np.array, cov: np.array, des_seq: tuple,
                 obs_seq: tuple, sigma: float) -> tuple:
    """
    Calculates the posterior mean and posterior covariance matrix after a
    batch update of multiple experiments.

    Keyword arguments:
    mean -- the prior mean matrix, size 2x2
    cov -- the prior covariance matrix, size 2x2
    des_seq -- the design sequence
    obs_seq -- the ordered observations across all experiments
    sigma -- the noise factor in the Bayesian update formula
    """
    for design, obs in zip(des_seq, obs_seq):
        design_mat = np.array([[1.0], [design]])
        mean, cov = bayesian_update(mean, cov, design_mat, obs, sigma)
    return mean, cov
