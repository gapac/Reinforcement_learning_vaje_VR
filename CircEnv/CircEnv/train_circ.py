import gym
import numpy as np
#from sb3_contrib import TQC
from stable_baselines3 import PPO
import os
import circ_env

env = gym.make('circ_env/Circle-v0')

# tensorboard logiranje
models_dir = "models/PPO01"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
logdir = "logs"
if not os.path.exists(logdir):  
    os.makedirs(logdir)


# PPO01
model = PPO('MlpPolicy', env=env, tensorboard_log=logdir, verbose=1, n_steps=2048,   batch_size=64,   gae_lambda=0.95,   gamma=0.99,  n_epochs=10,  ent_coef=0.0,  learning_rate=2.5e-4,  clip_range=0.2)
     
# Reset the environment
obs = env.reset()

# iteracija skozi učenje in shranjevanje modela
TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,tb_log_name="PPO_01")

    model.save(f"{models_dir}/{TIMESTEPS*iters}")
