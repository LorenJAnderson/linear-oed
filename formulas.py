import numpy as np
import scipy.linalg as lin


def calc_exp_kl(posterior, prior):
    return 0.5 * (np.log(lin.det(prior)) - np.log(lin.det(posterior)))


def bayesian_update(mean, cov, design_mat, obs, sigma):
    a_inv = lin.inv(lin.inv(cov) + (sigma ** (-2.0) * np.dot(design_mat, np.transpose(design_mat))))
    b = np.dot(lin.inv(cov), mean) + (sigma ** (-2.0) * design_mat * obs)
    return np.dot(a_inv, b), a_inv


def batch_update(mean, cov, designs, obs, sigma):
    for design in designs:
        design_mat = np.array([[1], [design]])
        mean, cov = bayesian_update(mean, cov, design_mat, obs, sigma)
    return mean, cov


def big_kl_div(designs, obs=1, sigma=1):
    mean = np.array([[0], [0]])
    cov = np.array([[1, 0], [0, 1]])
    post_mean = mean.copy()
    post_cov = cov.copy()
    for design in designs:
        design_mat = np.array([[1], [design]])
        post_mean, post_cov = bayesian_update(post_mean, post_cov, design_mat, obs, sigma)
    return calc_exp_kl(post_cov, cov)
