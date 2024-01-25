import environment

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

train_env = environment.Environment()
eval_env = environment.Environment()

eval_callback = EvalCallback(eval_env, log_path="logs", eval_freq=200,
                             n_eval_episodes=10, deterministic=True,
                             render=False)


model = DQN("MlpPolicy", train_env, verbose=1, buffer_size=10_000,
            learning_rate=0.001, target_update_interval=1_000,
            exploration_fraction=0.25, tensorboard_log="logs")

model.learn(total_timesteps=200_000, progress_bar=True, callback=eval_callback)
