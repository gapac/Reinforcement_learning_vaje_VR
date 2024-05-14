#Incializacija okolja in učenja

import gym
from stable_baselines3 import DQN 
import torch

print(torch.cuda.is_available())

env = gym.make("MountainCar-v0")

#Preverimo prostor akcij in opazovanja
print( "Actions = ", env.action_space.n)
print( "Obs space high = ", env.observation_space.high)
print( "Obs space low", env.observation_space.low)
# Inicializiramo učenje agenta
# Podatki za inicializacijo so na spletni strani ter na rl-baselines3-zoo
policy_kwargs = dict(net_arch=[256, 256])
model = DQN('MlpPolicy', 
            env=env,
            learning_rate=4e-3,
            batch_size=128,
            buffer_size=10000,
            learning_starts=1000,
            gamma=0.99,
            target_update_interval=600,
            train_freq=16,
            gradient_steps=8,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            policy_kwargs=policy_kwargs,
            seed=2,
            verbose=1,
            device='cuda'  # Specify 'cuda' to use GPU
)



#Učenje agenta
model.learn(total_timesteps=1.2e5)

#Shranimo model
model.save("dqn_car")