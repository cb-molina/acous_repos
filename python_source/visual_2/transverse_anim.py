# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:54:05 2020

@author: mrosew98
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#%%

def create_grid_pos(x_size, y_size):

    x_pos = np.zeros((y_size, x_size))
    y_pos = np.zeros((y_size, x_size))

    for x in range(x_size):
        y_pos[:,x] = np.arange(y_size)

    for y in range(y_size):
        x_pos[y,:] = np.arange(x_size)

    return x_pos, y_pos


def transverse_wave(x_positions, y_positions, frames_per_cycle=20, num_frames=100, wave_len=10, amplitude=1, phase = 0):
    # frames_per_cycle = number of frames per cycle
    # num_frames = frames to be generated
    k = 2 * np.pi / wave_len
    omega = 2 * np.pi / frames_per_cycle 
    x_frames = np.zeros((num_frames, x_positions.shape[0], x_positions.shape[1]))
    ## x_pos.shape: (5, 20), x_pos.shape[0]: 5, x_pos.shape[1]: 20
    y_frames = np.zeros((num_frames, y_positions.shape[0], y_positions.shape[1]))
    ## y_pos.shape: (5, 20), y_pos.shape[0]: 5, y_pos.shape[1]: 20

    for t in range(num_frames):
        x_frames[t,:,:] = x_positions
        for x in range(x_positions.shape[1]):
            y_frames[t,:,x] = y_positions[:,x] + amplitude * np.sin(k * x_positions[:,x] - omega * t + phase)
 
    return x_frames, y_frames
#%%
    
x_size = 20
y_size = 5
x_pos, y_pos = create_grid_pos(x_size=x_size, y_size=y_size)

frames_per_cycle = 100
num_frames = 400
x_trans, y_trans = transverse_wave(x_pos, y_pos, frames_per_cycle=frames_per_cycle, num_frames=num_frames, wave_len=10, amplitude=1, phase = 0)
#%%

fig_trans, ax_trans = plt.subplots()
fig_trans.set_size_inches(6,2.5)

j = 0
def animate(i):
    global j
    j += 1
    ax_trans.clear()
    ax_trans.scatter(x_trans[j,:,:], y_trans[j,:,:], color="cyan")
    plt.axis('off')
    
ani_trans = animation.FuncAnimation(fig_trans, animate, frames=num_frames, interval=50)
plt.show()
#%%
