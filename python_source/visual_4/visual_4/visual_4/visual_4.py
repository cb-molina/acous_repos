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
# k_1,2 values
k_1 = 1
k_2 = 3 * k_1


#x axis
x0 = np.linspace(0,pi,10000)

#Initial time
t0 = 0

#Time increment
dt = 0.05

def omega(k,v):
    return k * v

#Wave equation solution
def f(x,t):
    return np.sin(k_1 * x) * np.cos(omega(k_1,v) * t) + np.sin(k_2 * x) * np.cos(omega(k_2,v)* t)

def f1(x,t):
    return np.sin(k_1 * x) * np.cos(omega(k_1,v) * t)

def f2(x,t):
    return np.sin(k_2 * x) * np.cos(omega(k_2,v) * t)

a = []
b = []
c = []

for i in range(500):
    value = f(x0,t0)
    value1 = f1(x0,t0)
    value2 = f2(x0,t0)
    t0 = t0 + dt
    a.append(value)
    b.append(value1)
    c.append(value2)

k = 0
def animate(i):
    global k
    x = a[k]
    x1 = b[k]
    x2 = c[k]
    k += 1
    ax1 = plt.subplot(1,2,1)    
    ax1.clear()
    plt.plot(x0,x,color='#87FF59')
    plt.grid(True)
    plt.ylim([-2,2])
    plt.xlim([0,pi/k_1])

    ax2 = plt.subplot(2,2,2)
    ax2.clear()
    plt.plot(x0,x1,color='yellow')
    plt.grid(True)
    plt.ylim([-2,2])
    plt.xlim([0,pi/k_1])

    ax3 = plt.subplot(2,2,4)
    ax3.clear()
    plt.plot(x0,x2,color='orange')
    plt.grid(True)
    plt.ylim([-2,2])
    plt.xlim([0,pi/k_1])
    
anim = animation.FuncAnimation(fig,animate,frames=360,interval=20)
writer = PillowWriter(fps=25) 
# anim.save('waveeq2.gif', writer=writer)

plt.show()
