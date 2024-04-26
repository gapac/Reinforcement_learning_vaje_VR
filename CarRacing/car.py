import gym
import numpy as np
import random
from time import sleep
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make("CarRacing-v1", domain_randomize=True)
# normal reset, this changes the colour scheme by default
env.reset()
env.render()

#Prikaz parametrov
print( "Observation space = ", env.observation_space.n)
print( "Actions = ", env.action_space.n)

#deklaracija Q tabele lahko je zacetno stanje samo nule, lahk je pa random
q_table = np.zeros([env.observation_space.n, env.action_space.n])                                     #ALI
#q_table = np.random.uniform(low=0, high=1, size=[env.observation_space.n, env.action_space.n])       #ALI
print( "Q table size = ", q_table.shape)

#deklaracija parametrov
learning_rate = 0.1        #
discount_factor = 0.95     #
epochs = 10000             #

# Exploration parameters, za pogledat kk nastimat parametre uporabi epsilon.py
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.001            # Exponential decay rate for exploration prob
'''BOL BUTAST NACIN EPSILONA
#ta epsilon spreminjamo skozi cas
epsilon = 1                # Preklaplanje med dvema strategijama - raziskovanje in pozresnaMeoda
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = epochs//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
'''
SHOW_EVERY = 1000

#Začnemo z učenjem z izbranim številom epoh
for episode in range(epochs):
    state = env.reset()
    done = False
    trial_length = 0 #kolko korakov je potreboval do konca epizode

    # Izvedemo posamezni „sprehod čez okolje“ z izbiro akcij
        # raziskovanje: naključna akcija
        # uporabo zbranega znanja: akcija z maximalno q vrednostjo
    while not done:

        if (random.uniform(0, 1) < epsilon): # Exploration with random action
            action = env.action_space.sample()
        else:                                # Use the action with the highest q-value
            action = np.argmax(q_table[state]) 

        #Izvedemo akcijo v okolju in preberemo vrednosti novega stanja in nagrade
        next_state, reward, done, info = env.step(action)
        if (trial_length > 100):
            done=True

        #Posodobimo stanje Q tabele, zdej vemo kam nas je akcija pripeljala.
        curr_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state]) # ne zanima nas akcija ampak najvecja vrednost Q, ta vrednost nam pove kolko smo blizu cilja
        new_q = (1 - learning_rate) * curr_q + learning_rate * (reward + discount_factor * next_max_q) #GlAVNA ENACBA https://janezpodfe.github.io/VR_RL/images/Picture7.png
        q_table[state, action] = new_q #novo vrednost zapisemo v tabelo

        #Posodobimo stanje
        state = next_state

        #Shranimo dolžino trenutnega „sprehoda“
        if episode % SHOW_EVERY == 0:
            trial_length += 1

        if episode % SHOW_EVERY == 0:
            print(f'Episode: {episode:>5d}, episode length: {int(trial_length):>5d}')

        '''BOLJ BUTAST NACIN EPSILONA
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            #vsako epizodo zamnjsamo faktor epsilon, vedno manj je metode raziskovanja
            epsilon -= epsilon_decay_value
        '''
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


#Testiramo naučeno Q tabelo oziroma agenta
print('Q Table: ')
print(q_table)
state = env.reset()
env.render()
done = False
trial_length = 0

#TEST   -   Izvedemo sprehod skozi epizodo   -   pogledamo kaj se je nauco
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    trial_length += 1
    print(" Step " + str(trial_length))
    env.render()
    sleep(.2)
'''#Več testov, da vidimo uspešnost
lengths=[]
for trialnum in range(1, 11):
    state = env.reset()
   
    done = False
    trial_length = 0
    
    while not done and trial_length < 25:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        print("Trial number " + str(trialnum) + " Step " + str(trial_length))
        env.render()
        sleep(.2)
        trial_length += 1
    lengths.append(trial_length)
    
    sleep(.2)
avg_len=sum(lengths)/10
print(avg_len)

'''



'''MATPLOTLIB PRIKAZ-------------------------------------------------------------------------------------------------------'''
fig1, ax1 = plt.subplots()
ax1.axis('off')
ax1.axis('tight')

okolje = [["o","o","o","o","o","o","o","o","o","o","o","o","o","o","o","o"], ["o","o","o","o","o","o","o","o","o","o","o","o","o","o","o","o"], ["o","o","o","o","o","o","o","o","o","o","o","o","o","o","o","o"], ["S","x","x","x","x","x","x","x","x","x","x","x","x","x","x","G"]]

okolje_colours = np.asarray(okolje,dtype='U25')
okolje_rows = len(okolje[0][:])
okolje_columns = len(okolje[:][0])
okolje_colours[okolje_colours == "x"] = "firebrick"
okolje_colours[okolje_colours == "G"] = "gold"
okolje_colours[okolje_colours == "S"] = "limegreen"
okolje_colours[okolje_colours == "o"] = "cornflowerblue"

okolje = [["o,1","o","o","o","o","o","o","o","o","o","o","o","o","o","o","o,12"], ["o,13","o","o","o","o","o","o","o","o","o","o","o","o","o","o","o,24"], ["o,25","o","o","o","o","o","o","o","o","o","o","o","o","o","o","o,36"], ["S,37","x","x","x","x","x","x","x","x","x","x","x","x","x","x","G,48"]]

table_okolje = ax1.table(cellText = okolje,cellColours=okolje_colours, loc = 'center', cellLoc = 'center', rowLoc = 'center',colLoc = 'center')
table_okolje.scale(1,1.5)
table_okolje.auto_set_font_size(True)
table_okolje.set_fontsize(10)

#plt.show()
#print(q_table)

fig2, ax2 = plt.subplots()
fig2.patch.set_visible(False)
ax2.axis('off')
columns = ["GOR","DESNO","DOL","LEVO"]
rows = ["STANJE %d" %(i+1) for i in range(env.observation_space.n)]
qtable = np.around(q_table,3)
qtable_s = qtable[:][0:22]
rows_s = rows[0:22]
norm = plt.Normalize(qtable_s.min(), qtable_s.max()+0.1)
colours = plt.cm.YlGn(norm(qtable_s))
table = ax2.table(cellText=qtable_s, rowLabels=rows_s, colLabels=columns, loc = 'center', cellColours=colours,cellLoc ='center',rowLoc='center', colLoc ='center',colWidths=[0.1,0.1,0.1,0.1,0.1])
table.auto_set_font_size(False)
table.set_fontsize(8)

fig3, ax3 = plt.subplots()
fig3.patch.set_visible(False)
ax3.axis('off')
qtable_s = qtable[:][23:48]
rows_s = rows[23:48]
colours = plt.cm.YlGn(norm(qtable_s))
table = ax3.table(cellText=qtable_s, rowLabels=rows_s, colLabels=columns, loc = 'center', cellColours=colours,cellLoc ='center',rowLoc='center', colLoc ='center',colWidths=[0.1,0.1,0.1,0.1,0.1])
table.auto_set_font_size(False)
table.set_fontsize(8)

plt.show()


