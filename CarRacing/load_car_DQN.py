#Incializacija okolja in učenja

import gym
from stable_baselines3 import DQN 
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("MountainCar-v0")

print( "Actions = ", env.action_space.n)
print( "Obs space high = ", env.observation_space.high)
print( "Obs space low", env.observation_space.low)
#Naložimo in incializiramo agenta

#model = DQN.load("dqn_car", env=env)

#izbira najbolsega agenta ki smo ga najdli z tenzorflov
model = DQN.load("100000", env=env)
#Zaženemo in testiramo agenta

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward}, Std reward: {std_reward}')

obs = env.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()