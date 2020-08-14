# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:54:05 2020

@author: mrosew98
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
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

frames_per_cycle = 100
num_frames = 400

#number of particles/data points on the graph
num_particles = 36
#array of x inputs with shape=(num_particles, 1)
x_inputs_trans = np.array([np.linspace(0, 4*np.pi, num_particles)])

#get data points
x_trans, y_trans = transverse_wave(x_inputs_trans, frames_per_cycle=frames_per_cycle,
                                   num_frames=num_frames, wave_len=2*np.pi, amplitude=1, phase=0)

#get data points
#x_long, y_long = longitudinal_wave(x_pos, y_pos, frames_per_cycle=frames_per_cycle,
#                                   wave_len=10, amplitude=1, num_frames=num_frames )
#%%

fig_waves = plt.figure()
fig_waves.set_size_inches(6,2.5)
ax_trans = fig_waves.add_subplot(111)
#ax_trans = fig_waves.add_subplot(2,1,1)
#ax_test = fig_waves.add_subplot(3,1,2)
#ax_long = fig_waves.add_subplot(2,1,2)

#setup for vertical line
x_phi_v = np.repeat(-1.5, num_frames)
y_phi_min_v = np.repeat(0.0, num_frames)
y_phi_max_v = np.repeat(1.0, num_frames)
#setup for horizontal line (top)
y_phi_h1 = np.repeat(1.0, num_frames)
x_phi_min_h1 = np.repeat(-1.5, num_frames)
x_phi_max_h1 = np.repeat(-1.0, num_frames)
#setup for horizontal line (bottom)
y_phi_h2 = np.repeat(0.00, num_frames)
x_phi_min_h2 = np.repeat(-1.5, num_frames)
x_phi_max_h2 = np.repeat(-1.0, num_frames)
#setup for text
text_x = np.repeat(-2.8, num_frames)
text_y = np.repeat(0.35, num_frames)
#horizontal line at equilibrum
zero_line_y = np.repeat(0.0, num_frames)


j = 0
def animate(i):
    global j
    j += 1
   
    ax_trans.clear()
    #ax_trans.set_axis_off()
    ax_trans.scatter(x_trans[j,:,:], y_trans[j,:,:], color="cyan")
    ax_trans.scatter(x_trans[j,0,0], y_trans[j,0,0], color="purple")
    #ax_trans.scatter(x_trans[j,0,-1], y_trans[j,0,-1], color="purple")
    plt.xlim(-np.pi,4*np.pi)
    plt.ylim(-1.5,1.5)
    ## marker to show phi
    plt.vlines(x_phi_v[j], y_phi_min_v[j], y_phi_max_v[j], colors="black", linestyles="solid", linewidth=2)
    plt.hlines(y_phi_h1[j], x_phi_min_h1[j], x_phi_max_h1[j], colors="black", linestyles="solid", linewidth=2)
    plt.hlines(y_phi_h2[j], x_phi_min_h2[j], x_phi_max_h2[j], colors="black", linestyles="solid", linewidth=2)
    ## text on plot
    plt.text(text_x[j], text_y[j], "$\Phi$", fontsize=25)
    plt.axhline(zero_line_y[j], linestyle="dashed", alpha=0.4)
    
    ax_trans.axes.xaxis.set_ticklabels([])
    ax_trans.axes.yaxis.set_ticklabels([])
    ax_trans.axes.xaxis.set_ticks([])
    ax_trans.axes.yaxis.set_ticks([])

#    ax_long.clear()
#    ax_long.scatter(x_long[j,:,:], y_long[j,:,:], color="orange")
#    plt.xlim(0,20)
#    #plt.axis('off')

frames_per_seconds = 40
interval = 1000/frames_per_seconds

animation_waves = animation.FuncAnimation(fig_waves, animate, frames=num_frames, interval=interval)
#writer = PillowWriter(fps=25)
#animation_waves.save('visual_2.gif', writer=writer)
plt.show()


#%%