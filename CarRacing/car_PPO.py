import gym
from stable_baselines3 import PPO
import os

models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir) 

env = gym.make("MountainCar-v0")

policy_kwargs = dict(net_arch=[256, 256])
model = PPO('MlpPolicy', 
            env=env, 
            tensorboard_log=logdir,
            verbose=1,
            # learning_rate=4e-3,
            # batch_size=128,
            # buffer_size=10000,
            # gamma=0.98,
            # n_steps=2048,
            # gae_lambda=0.95,
            # ent_coef=0.0,
            # clip_range=0.2,
            # policy_kwargs=policy_kwargs
            )

TIMESTEPS = 2000
iters = 0
while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")