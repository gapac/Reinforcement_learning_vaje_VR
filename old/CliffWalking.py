import gym
import numpy as np
import random
from time import sleep

env = gym.make('CliffWalking-v0')

env.reset()
env.render()

print( "Observation space = ", env.observation_space.n)
print( "Actions = ", env.action_space.n)

q_table = np.zeros([env.observation_space.n, env.action_space.n])
#q_table = np.random.uniform(low=0, high=1, size=[env.observation_space.n, env.action_space.n])
print( "Q table size = ", q_table.shape)

#deklaracija parametrov
learning_rate = 0.5        #
discount_factor = 0.95     #
epochs = 10000             #
epsilon = 1                # Preklaplanje med dvema strategijama - exploracija in pozresnaMetoda

# ta epsilon skozi cas spreminjamo
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = epochs//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
SHOW_EVERY = 1000

for episode in range(epochs):
    state = env.reset()
    done = False
    trial_length = 0

    while not done:
        if (random.uniform(0, 1) < epsilon): # Exploration with random action
            action = env.action_space.sample()
        else: # Use the action with the highest q-value
            action = np.argmax(q_table[state]) 

        next_state, reward, done, info = env.step(action)

        curr_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * curr_q + learning_rate * (reward + discount_factor * next_max_q)
        q_table[state, action] = new_q

        state = next_state

        if episode % SHOW_EVERY == 0:
            trial_length += 1

    if episode % SHOW_EVERY == 0:
        print(f'Episode: {episode:>5d}, episode length: {int(trial_length):>5d}')

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


print(q_table)
state = env.reset()
env.render()
done = False
trial_length = 0

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    trial_length += 1
    print(" Step " + str(trial_length))
    env.render()
    sleep(.2)
