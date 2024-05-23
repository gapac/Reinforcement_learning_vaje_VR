import gym
import numpy as np
#from sb3_contrib import TQC
from stable_baselines3 import PPO
import os
import circ_env

env = gym.make('circ_env/Circle-v0', render_mode="human")

model = PPO.load("./models/PPO01/5320000", env=env)

# Reset the environment
obs = env.reset()

EPISODES = 1000

for episode in range(EPISODES):

    obs = env.reset()
    done = False

    while not done:

        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        env.render()
        
# Close the environment
env.close()