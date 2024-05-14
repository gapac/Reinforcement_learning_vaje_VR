import gym
import numpy as np
from time import sleep
import sys ###
import matplotlib.pyplot as plt ###
import matplotlib.patches as mpatches ###

env = gym.make("MountainCar-v0")

#DISCRETE_OS_SIZE = [20, 20]
GRID_SIZE = 20;
DISCRETE_OS_SIZE = [GRID_SIZE]*len(env.observation_space.high)

obs_high = env.observation_space.high
obs_low = env.observation_space.low
discrete_os_win_size = (obs_high - obs_low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)

def get_discrete_state(state):
    discrete_state = (state - obs_low)/discrete_os_win_size
    discrete_state = np.clip(discrete_state.astype(int),0,GRID_SIZE-1)
    return tuple(discrete_state)

q_table = np.load(f"car_e14999-qtable.npy") 
print( "Q table size = ", q_table.shape)

state = env.reset()
discrete_state = get_discrete_state(state)
env.render()
done = False
while not done:
    action = np.argmax(q_table[discrete_state])
    state, reward, done, info = env.step(action)
    discrete_state = get_discrete_state(state)
    env.render()
    #sleep(0.5)
env.close()

def get_actions(dataset):
    stolpec = 0
    #print(type(data))
    actions = np.ndarray([GRID_SIZE, GRID_SIZE])
    for stolpec in range(GRID_SIZE):
        vrstica = 0
        for vrstica in range(GRID_SIZE):
            if dataset[stolpec, vrstica, 0] == dataset[stolpec, vrstica, 1] == dataset[stolpec, vrstica, 2] == 0:
                actions[stolpec, vrstica] = -1
            else:
                actions[stolpec, vrstica] = np.argmax(dataset[stolpec, vrstica])
    return actions

def plot_graphs(ep_list):
    ep = 0
    fig,axs = plt.subplots(2,2,figsize = (15,15))
    fig.suptitle("Izbira akcije glede na Q tabelo za različno število epizod", fontsize = 16)
    print(enumerate(axs.flat))
    for i, ax in enumerate(axs.flat):
        data = np.load('car_e'+str(ep_list[ep])+'-qtable.npy')
        np.set_printoptions(threshold=sys.maxsize)

        positions = np.arange(-1.2,0.6+discrete_os_win_size[0],discrete_os_win_size[0])
        velocities = np.arange(-0.07,0.07+discrete_os_win_size[1],discrete_os_win_size[1])

        #ax = fig.add_subplot(2,2)
        labels = ["Neobiskana stanja","Premik levo", "Ne naredimo ničesar", "Premik desno"]
        cmap = plt.colormaps.get_cmap('Blues') #matplotlib.colormaps

        ax = plt.subplot(2,2,i+1)
        actions = get_actions(data)

        ax.pcolor(velocities,positions, actions, cmap = cmap)
        ax.set_ylabel("Pozicija", fontsize = 14)
        ax.set_xlabel("Hitrost", fontsize = 14)        
        #ax.hlines(y=0, xmin=-0.6, xmax=-0.4, linewidth=2, color='r')
        #ax.plot(velocities_0[0],positions_0[0],'ro')
        ax.set_title('Epizoda '+str(ep_list[ep]+1), fontsize = 13)
        ep += 1

    bound = np.linspace(0, 1, 5)
    print(bound)
    fig.legend([mpatches.Patch(color=cmap(b)) for b in bound[:-1]],
               [labels[i] for i in range(4)], loc = 'upper right')
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    fig.savefig('Qtable.jpg') ###
    plt.show()

plot_graphs([0,5000,10000,14000])