import gym
import numpy as np

# Initialize the "Cart-Pole" environment
env = gym.make('LunarLander-v2')

# Define the size of the Q-table
state_size = (1, 1, 1, 1, 1, 1, 1, 1)
action_size = env.action_space.n

# Initialize Q table
Q = np.zeros(state_size + (action_size,))

# Define hyperparameters
alpha = 0.5
gamma = 0.6
epsilon = 0.1
episodes = 50000

# Q-Learning algorithm
for i_episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(Q[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = Q[state][action]
        next_max = np.max(Q[next_state])

        # Update Q-value for current state
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q[state][action] = new_value

        state = next_state

    if i_episode % 100 == 0:
        print(f"Episode: {i_episode}")

env.close()