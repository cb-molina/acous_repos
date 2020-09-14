""" Notes on this version: visual_4_v2
For the project, utilization of this version of visual_4 has been lost and may be changed entirely
But this code is really good in my personal opinion, for it utilizes various defined functions and shows different modes
of a wave with it's time evolution.

Christian B. Molina 08/27/2020
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation

plt.style.use('dark_background') # background style

fig = plt.figure(figsize=(12,6)) # creates a figure object
fig.set_dpi(100)            

#Wave speed
v = 1
# k values
k_1 = 1
k_values = [k_1, 3*k_1, 9*k_1, 27*k_1]
#x axis
x0 = np.linspace(0,pi,10000)
#Initial time
t0 = 0
#Time increment
dt = 0.05


def omega(k,v):
    return k * v
# Wave Equation - time part
def g(t,n):
    return np.cos(omega(k_values[n],v) * t)
# Wave Equation - spatial part
def f(x,n):
    return np.sin(k_values[n] * x)

a = []

def genWave(n):
    global dt
    global t0
    global x0
    for i in range(100):
        for m in range(-1,n):
            value = f(x0,n) * g(t0,n)
        t0 = t0 + dt
        a.append(value)

#Execution of triangle wave
genWave(0)
genWave(1)
genWave(2)
#genWave(3)

k = 0
def animate(i):
    global k
    x = a[k]
    k += 1
    ax1 = plt.subplot(1,2,1)    
    ax1.clear()
    plt.plot(x0,x,color='#87FF59')
    plt.grid(True)
    plt.ylim([-2,2])
    plt.xlim([0,pi/k_1])
    
anim = animation.FuncAnimation(fig,animate,frames=360,interval=20)
writer = PillowWriter(fps=25) 
#anim.save('visual_4.gif', writer=writer)

plt.show()
