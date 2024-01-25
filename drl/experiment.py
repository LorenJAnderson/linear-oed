import copy

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

import formulas


class LinearOEDEnv(gym.Env):
    """
    Linear OED environment.

    Attributes:
    action_space -- the gymnasium-standard action space
    observation_space -- the gymnasium-standard observation space
    time_step -- the current time step of the environment
    state -- the current state of the environment
    """
    def __init__(self) -> None:
        """Initializes linear OED environment."""
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))
        self.time_step = None
        self.state = None

    def reset(self, seed=None, options=None) -> tuple:
        """
        Resets environment fields and returns copy of state and info
        dictionary.
        """
        self.time_step = 0
        self.state = np.array([0.0]*10)
        info = {}
        return copy.deepcopy(self.state), info

    def step(self, action: int) -> tuple:
        """Takes action and produces gymnasium-standard environment signals."""
        design = 0.1 + 0.1*action
        self.state[self.time_step] = design
        self.time_step += 1
        term = self.time_step == 10
        trunc = False
        reward = 0
        if term:
            prior_mean = np.array([[0.0], [0.0]])
            prior_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
            obs = tuple([1.0] * 10)
            post_mean, post_cov = formulas.batch_update(
                prior_mean, prior_cov, self.state, obs, sigma=1.0)
            reward = formulas.calc_exp_kl(post_cov, prior_cov)
        info = {}
        return copy.deepcopy(self.state), reward, term, trunc, info


def experiment() -> None:
    """
    Trains DQN agent on linear OED environment. Saves tensorboard log for
    training rewards and saves evaluations log for testing rewards.
    Evaluation is deterministic, so only one evaluation episode is needed.
    """
    train_env = LinearOEDEnv()
    eval_env = LinearOEDEnv()
    eval_callback = EvalCallback(eval_env, log_path="logs", eval_freq=200,
                                 n_eval_episodes=1, deterministic=True,
                                 render=False)
    model = DQN("MlpPolicy", train_env, verbose=1, buffer_size=10_000,
                learning_rate=0.001, target_update_interval=1_000,
                exploration_fraction=0.25, tensorboard_log="logs")
    model.learn(total_timesteps=200_000, progress_bar=True,
                callback=eval_callback)


if __name__ == "__main__":
    experiment()

