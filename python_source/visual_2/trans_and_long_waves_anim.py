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
    
#    for x in range(x_size):
#        y_pos[:,x] = np.linspace(0, y_size, 5)

    for y in range(y_size):
        x_pos[y,:] = np.arange(x_size)
        
#    for y in range(y_size):
#        x_pos[y,:] = np.linspace(0, x_size, 10)

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


def longitudinal_wave(x_positions, y_positions, frames_per_cycle=20, wave_len=10, amplitude=1, phase = 0, num_frames=100):
    # frames_per_cycle = number of frames per cycle
    # num_frames = frames to be generated
    k = 2 * np.pi / wave_len
    omega = 2 * np.pi / frames_per_cycle 
    x_frames = np.zeros((num_frames, x_positions.shape[0], x_positions.shape[1]))
    y_frames = np.zeros((num_frames, y_positions.shape[0], y_positions.shape[1]))

    for t in range(num_frames):
        y_frames[t,:,:] = y_positions
        for x in range(x_positions.shape[1]):
            x_frames[t,:,x] = x_positions[:,x] + amplitude * np.sin(k * x_positions[:,x] - omega * t + phase)

    return x_frames, y_frames
#%%
    
x_size = 20
y_size = 5
x_pos, y_pos = create_grid_pos(x_size=x_size, y_size=y_size)

print("x_pos:", x_pos, "\n", "y_pos:", y_pos)
print("x_pos.shape:", x_pos.shape, "\n", "y_poss.shape:", y_pos.shape)
#%%
frames_per_cycle = 100
num_frames = 800

x_trans, y_trans = transverse_wave(x_pos, y_pos, frames_per_cycle=frames_per_cycle,
                                   num_frames=num_frames, wave_len=10, amplitude=1, phase = 0)

x_long, y_long = longitudinal_wave(x_pos, y_pos, frames_per_cycle=frames_per_cycle,
                                   wave_len=10, amplitude=1, num_frames=num_frames )
#%%

fig_waves = plt.figure()
fig_waves.set_size_inches(8,6)
ax_trans = fig_waves.add_subplot(111)
#ax_trans = fig_waves.add_subplot(2,1,1)
#ax_long = fig_waves.add_subplot(2,1,2)

#setup for vertical line
x_phi_v = np.repeat(-1.5, num_frames)
y_phi_min_v = np.repeat(-1.0, num_frames)
y_phi_max_v = np.repeat(0.98, num_frames)
#setup for horizontal line (top)
y_phi_h1 = np.repeat(0.98, num_frames)
x_phi_min_h1 = np.repeat(-2.0, num_frames)
x_phi_max_h1 = np.repeat(-1.0, num_frames)
#setup for horizontal line (bottom)
y_phi_h2 = np.repeat(-1.0, num_frames)
x_phi_min_h2 = np.repeat(-2.0, num_frames)
x_phi_max_h2 = np.repeat(-1.0, num_frames)

j = 0
def animate(i):
    global j
    j += 1
   
    ax_trans.clear()
    #ax_trans.set_axis_off()
    ax_trans.scatter(x_trans[j,:,:], y_trans[j,:,:], color="cyan")
    ## tracking particle
    ax_trans.scatter(x_trans[j,0,0], y_trans[j,0,0], color="purple")
    ## marker to show phi
    plt.vlines(x_phi_v[j], y_phi_min_v[j], y_phi_max_v[j], colors="black", linestyles="solid")
    plt.hlines(y_phi_h1[j], x_phi_min_h1[j], x_phi_max_h1[j], colors="black", linestyles="solid")
    plt.hlines(y_phi_h2[j], x_phi_min_h2[j], x_phi_max_h2[j], colors="black", linestyles="solid")
    plt.ylim(-2,6)

#    ax_long.clear()
#    ax_long.set_axis_off()
#    ax_long.scatter(x_long[j,:,:], y_long[j,:,:], color="orange")
#    plt.xlim(0,20)


frames_per_seconds = 40
interval = 1000/frames_per_seconds

#print(np.min(y_trans[:,0,0]))

animation_waves = animation.FuncAnimation(fig_waves, animate, frames=num_frames, interval=interval)
plt.show()
#animation_waves.save(os.path.join('..', 'output', 'C1', 'name.mp4'))
#%%