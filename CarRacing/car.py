import gym
import numpy as np
from time import sleep
import matplotlib.pyplot as plt ###


env = gym.make("MountainCar-v0")
#env = gym.make("Acrobot-v1")
#env = gym.make("CartPole-v1")

LEARNING_RATE = 0.2

DISCOUNT = 0.95
EPISODES = 15000
SHOW_EVERY = 1000

epsilon = 1.0 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

print( "Actions = ", env.action_space.n)
print( "Obs space high = ", env.observation_space.high)
print( "Obs space low", env.observation_space.low)

#DISCRETE_OS_SIZE = [20, 20]
GRID_SIZE = 20
DISCRETE_OS_SIZE = [GRID_SIZE]*len(env.observation_space.high)

obs_high = env.observation_space.high
obs_low = env.observation_space.low
discrete_os_win_size = (obs_high - obs_low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)

#q_table = np.random.uniform(low=-1, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
q_table = np.zeros(DISCRETE_OS_SIZE + [env.action_space.n])
print( "Q table size = ", q_table.shape)

def get_discrete_state(state):
    discrete_state = (state - obs_low)/discrete_os_win_size
    discrete_state = np.clip(discrete_state.astype(int),0,GRID_SIZE-1)
    return tuple(discrete_state)

reward_list = [] ###
ave_reward_list = [] ###

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False
    tot_reward, reward = 0, 0 

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        

        if episode % SHOW_EVERY == 0:
            env.render()

        if not done:

            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state   
        
        tot_reward += reward 

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    if episode % SHOW_EVERY == 0:
        #save to folder CarRacing
        np.save(f"car_e{episode}-qtable.npy", q_table)

    reward_list.append(tot_reward) ###
    if episode % SHOW_EVERY//10 == 0: ###
        ave_reward = np.mean(reward_list) ###
        ave_reward_list.append(ave_reward) ###
        reward_list = [] ###
        print('Episode {} Average Reward: {}'.format(episode, ave_reward)) ###

#shranimo se zadnjo epizodo
np.save(f"car_e{episode}-qtable.npy", q_table) 
# Plot Rewards
fig, ax = plt.subplots()
ax.plot(SHOW_EVERY * (np.arange(len(ave_reward_list)) + 1), ave_reward_list) ###
ax.set_xlabel('Episodes') ###
ax.set_ylabel('Average Reward') ###
ax.set_title('Average Reward vs Episodes') ###
fig.savefig('rewards.jpg') ###
plt.show() ###