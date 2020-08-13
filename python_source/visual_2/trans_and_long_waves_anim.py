# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:54:05 2020

@author: mrosew98
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#%%

#function for transverse wave
def transverse_wave(x, frames_per_cycle, num_frames, wave_len=2*np.pi, amplitude=1, phase=0):

    k = 2 * np.pi / wave_len
    omega = 2 * np.pi / frames_per_cycle 
    x_frames = np.zeros((num_frames, x.shape[0], x.shape[1]))
    y_frames = np.zeros((num_frames, x.shape[0], x.shape[1]))

    for t in range(num_frames):
        x_frames[t,:,:] = x
        for i in range(x.shape[1]):
            y_frames[t,:,i] = amplitude * np.sin(k * x[:,i] - omega*t + phase)
 
    return x_frames, y_frames

#def longitudinal_wave(x_positions, y_positions, frames_per_cycle=20, wave_len=10, amplitude=1, phase = 0, num_frames=100):
#    # frames_per_cycle = number of frames per cycle
#    # num_frames = frames to be generated
#    k = 2 * np.pi / wave_len
#    omega = 2 * np.pi / frames_per_cycle 
#    x_frames = np.zeros((num_frames, x_positions.shape[0], x_positions.shape[1]))
#    y_frames = np.zeros((num_frames, y_positions.shape[0], y_positions.shape[1]))
#
#    for t in range(num_frames):
#        y_frames[t,:,:] = y_positions
#        for x in range(x_positions.shape[1]):
#            x_frames[t,:,x] = x_positions[:,x] + amplitude * np.sin(k * x_positions[:,x] - omega * t + phase)
#
#    return x_frames, y_frames
#%%

#number of particles/data points on the graph
num_particles = 48
#array of x inputs with shape=(num_particles, 1)
x_inputs = np.array([np.linspace(0, 6*np.pi, num_particles)])

frames_per_cycle = 100
num_frames = 400

x_trans, y_trans = transverse_wave(x_inputs, frames_per_cycle=frames_per_cycle,
                                   num_frames=num_frames, wave_len=2*np.pi, amplitude=1, phase=0)

#x_long, y_long = longitudinal_wave(x_pos, y_pos, frames_per_cycle=frames_per_cycle,
#                                   wave_len=10, amplitude=1, num_frames=num_frames )
#%%

fig_waves = plt.figure()
fig_waves.set_size_inches(6,2.5)
ax_trans = fig_waves.add_subplot(111)
#ax_trans = fig_waves.add_subplot(2,1,1)
#ax_test = fig_waves.add_subplot(3,1,2)
#ax_long = fig_waves.add_subplot(2,1,2)

j = 0
def animate(i):
    global j
    j += 1
   
    ax_trans.clear()
    ax_trans.scatter(x_trans[j,:,:], y_trans[j,:,:], color="cyan")
    ax_trans.scatter(x_trans[j,0,0], y_trans[j,0,0], color="purple")
    ax_trans.scatter(x_trans[j,0,-1], y_trans[j,0,-1], color="purple")
    plt.xlim(0,6*np.pi)
    plt.ylim(-1,1)

#    ax_long.clear()
#    ax_long.scatter(x_long[j,:,:], y_long[j,:,:], color="orange")
#    plt.xlim(0,20)
#    #plt.axis('off')

frames_per_seconds = 40
interval = 1000/frames_per_seconds

animation_waves = animation.FuncAnimation(fig_waves, animate, frames=num_frames, interval=interval)
plt.show()
#animation_waves.save(os.path.join('..', 'output', 'C1', 'name.mp4'))
#%%