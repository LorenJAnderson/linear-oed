import copy

import numpy as np
import gymnasium as gym

import formulas


class Environment(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))
        self.time_step = None
        self.state = None

    def reset(self, seed=None, options=None):
        """Resets environment fields and returns"""
        self.time_step = 0
        self.state = np.array([0.0]*10)
        info = {}
        return copy.deepcopy(self.state), info

    def step(self, action):
        """ #TODO """
        converted_action = action*0.1 + 0.1
        self.state[self.time_step] = converted_action
        self.time_step += 1
        term = self.time_step == 10
        trunc = False
        reward = 0
        if term:
            mean = np.array([[0.0], [0.0]])
            prior_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
            obs = tuple([1.0] * 10)
            post_mean, post_cov = formulas.batch_update(
                mean, prior_cov, self.state, obs, sigma=1.0)
            reward = formulas.calc_exp_kl(post_cov, prior_cov)
        info = {}
        return copy.deepcopy(self.state), reward, term, trunc, info







