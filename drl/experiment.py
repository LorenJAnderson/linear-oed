import environment

from stable_baselines3 import DQN

env = environment.Environment()

model = DQN("MlpPolicy", env, verbose=1, buffer_size=10_000,
            learning_rate=0.001, target_update_interval=1_000,
            exploration_fraction=0.25)
model.learn(total_timesteps=200_000, progress_bar=True)
