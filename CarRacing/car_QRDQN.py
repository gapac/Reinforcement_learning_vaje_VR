import gym
from sb3_contrib import QRDQN
import os

models_dir = "models/QRDQN"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir) 

env = gym.make("MountainCar-v0")

policy_kwargs = dict(net_arch=[256, 256], n_quantiles=25)
model = QRDQN('MlpPolicy', 
            env=env, 
            tensorboard_log=logdir,
            verbose=1,
            learning_rate=4e-3,
            batch_size=128,
            buffer_size=10000,
            learning_starts=1000,
            gamma=0.98,
            target_update_interval=600,
            train_freq=16,
            gradient_steps=8,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            policy_kwargs=policy_kwargs)

TIMESTEPS = 2000
iters = 0
while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="QRDQN")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")