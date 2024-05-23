#Incializacija okolja in učenja

import gym
from stable_baselines3 import A2C
import torch
import os

models_dir = "models/A2C"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir) 

print(torch.cuda.is_available())

env = gym.make("MountainCar-v0")

#Preverimo prostor akcij in opazovanja
print( "Actions = ", env.action_space.n)
print( "Obs space high = ", env.observation_space.high)
print( "Obs space low", env.observation_space.low)
# Inicializiramo učenje agenta
# Podatki za inicializacijo so na spletni strani ter na rl-baselines3-zoo
policy_kwargs = dict(net_arch=[256, 256])
# Initialize the model
model = A2C(
    policy='MlpPolicy',
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    tensorboard_log= logdir,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=0,
    device='auto',
    _init_setup_model=True
)



#Učenje agenta
#model.learn(total_timesteps=1.2e5)
#Shranimo model
#model.save("dqn_car")

#Drug nacin ucenja agenta
TIMESTEPS = 2000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

