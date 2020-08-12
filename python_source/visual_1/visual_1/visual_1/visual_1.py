# visual_1

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation

# Constants
i = 1
j = 1
a_ij = i * 1
L = 5
v = 1
k_i = i * np.pi / L 
k_j = j * np.pi / L
k_i2 = 3 * k_i 
k_j2 = 3 * k_j 
k_i3 = 3 * k_i2 
k_j3 = 3 * k_j2 
w_ij = np.sqrt(k_i ** 2 + k_j ** 2) * v

def f(x,y,t): # our function
    return a_ij * np.cos(w_ij * t) * np.sin(k_i * x) * np.sin(k_j * y) + a_ij * np.cos(w_ij * t) * np.sin(k_i2 * x) * np.sin(k_j2 * y) + a_ij * np.cos(w_ij * t) * np.sin(k_i3 * x) * np.sin(k_j3 * y)


# Points of generated data
x0 = np.linspace(0,pi,100)
y0 = np.linspace(0,pi,100)

#Initial time, time step
t0 = 0
dt = 0.05

# Meshgrid
X, Y = np.meshgrid(x0, y0)    #(1)

a = []
for i in range(500):
    value = f(X,Y,t0)
    t0 = t0 + dt
    a.append(value)

plt.contourf(X, Y, value, 100, cmap='Spectral')
plt.colorbar();

plt.show()


#k = 0
#def animate(i):
#    global k
#    x = a[k]
#    k += 1
#    ax1 = plt.subplot(1,2,1)    
#    ax1.clear()
#    plt.plot(x0,x,color='#87FF59')
#    plt.grid(True)
#    plt.ylim([-2,2])
#    plt.xlim([0,pi/k_1])

# ===========================================================================
# For animating:
#for i in range(500):
#    value = f(x0,t0)
#    value1 = f1(x0,t0)
#    value2 = f2(x0,t0)
#    t0 = t0 + dt
#    a.append(value)
#    b.append(value1)
#    c.append(value2)

#k = 0
#def animate(i):
#    global k
#    x = a[k]
#    x1 = b[k]
#    x2 = c[k]
#    k += 1
#    ax1 = plt.subplot(1,2,1)    
#    ax1.clear()
#    plt.plot(x0,x,color='#87ff59')
#    plt.grid(true)
#    plt.ylim([-2,2])
#    plt.xlim([0,pi/k_1])

#    ax2 = plt.subplot(2,2,2)
#    ax2.clear()
#    plt.plot(x0,x1,color='yellow')
#    plt.grid(true)
#    plt.ylim([-2,2])
#    plt.xlim([0,pi/k_1])

#    ax3 = plt.subplot(2,2,4)
#    ax3.clear()
#    plt.plot(x0,x2,color='orange')
#    plt.grid(true)
#    plt.ylim([-2,2])
#    plt.xlim([0,pi/k_1])
    
#anim = animation.funcanimation(fig,animate,frames=360,interval=20)
#writer = pillowwriter(fps=25) 
# anim.save('waveeq2.gif', writer=writer)

