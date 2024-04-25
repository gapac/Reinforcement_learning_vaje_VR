import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.001            # Exponential decay rate for exploration prob

total_episodes = 10000

def plot_fcn(min_epsilon_fcn, max_epsilon_fcn, decay_rate_fcn, total_episodes_fcn):
    epsilon_values = [(min_epsilon_fcn + (max_epsilon_fcn - min_epsilon_fcn) * np.exp(-decay_rate_fcn * episode)) for episode in range(total_episodes_fcn)]
    ep_num = range(total_episodes_fcn)
    return epsilon_values

#define inital values for sliders
init_min_epsilon = min_epsilon
init_max_epsilon = max_epsilon
init_decay_rate = decay_rate
init_total_episodes = total_episodes

fig1,ax1 = plt.subplots()
line, = plt.plot(plot_fcn(init_min_epsilon,init_max_epsilon,init_decay_rate,init_total_episodes), lw=2)
ax1.set_xlabel("Epizoda", fontsize = 15)
ax1.set_ylabel("Epsilon vrednost",fontsize = 15)
plt.subplots_adjust( bottom=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Vrednost parametra epsilon v odvisnosti od števila epizod (učenje Q)",fontsize = 18)
plt.gcf().text(0.4,0.05,r'$\epsilon = \epsilon_{min} + (\epsilon_{max} - \epsilon_{min})*e^{(\lambda*N)}$',fontsize=18)

#slider for decay rate
ax_decay_rate = plt.axes([0.25, 0.1, 0.65, 0.03])
decay_rate_slider = Slider(
    ax=ax_decay_rate,
    label='Lambda',
    valmin=0.0001,
    valmax=0.01,
    valinit=init_decay_rate,

)
#slider for max epsilon value
ax_max_epsilon= plt.axes([0.25, 0.13, 0.65, 0.03])
max_epsilon_slider = Slider(
    ax=ax_max_epsilon,
    label='Epsilon (max)',
    valmin=0.1,
    valmax=1,
    valinit=init_max_epsilon,
)
#slider for min epsilon value
ax_min_epsilon= plt.axes([0.25, 0.16, 0.65, 0.03])
min_epsilon_slider = Slider(
    ax=ax_min_epsilon,
    label='Epsilon (min)',
    valmin=0,
    valmax=0.5,
    valinit=init_min_epsilon,
)
#slider for max episode value
ax_total_episodes= plt.axes([0.25, 0.19, 0.65, 0.03])
total_episodes_slider = Slider(
    ax=ax_total_episodes,
    label='Skupno število epizod - N',
    valmin=1,
    valstep=1,
    valmax=10000,
    valinit=init_total_episodes,
)

def update(val):
    line.set_data(range(total_episodes_slider.val),plot_fcn(min_epsilon_slider.val,max_epsilon_slider.val, decay_rate_slider.val,total_episodes_slider.val))
    #ax3.set_xlim(0,total_episodes_slider.val)
    ax3.autoscale_view(True,True,True)
    ax3.relim()
    fig.canvas.draw_idle()

decay_rate_slider.on_changed(update)
max_epsilon_slider.on_changed(update)
min_epsilon_slider.on_changed(update)
total_episodes_slider.on_changed(update)
decay_rate_slider.label.set_size(16)
max_epsilon_slider.label.set_size(16)
min_epsilon_slider.label.set_size(16)
total_episodes_slider.label.set_size(16)


resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')
button.label.set_size(16)
def reset(event):
    decay_rate_slider.reset()
    max_epsilon_slider.reset()
    min_epsilon_slider.reset()
    total_episodes_slider.reset()
    ax3.autoscale_view(True, True, True)
    ax3.relim()
button.on_clicked(reset)
plt.show()