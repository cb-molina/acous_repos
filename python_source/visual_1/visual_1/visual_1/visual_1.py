# visual_1

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation

def f(x,y): # This is our function
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


# Points of generated data
x = np.linspace (0, 5, 50)
y = np.linspace (0, 5, 40)

# Meshgrid
print("before meshgrid: \n","x-values: \n",x,"\n\n y-values:\n",y,"\n")
X, Y = np.meshgrid(x, y)    #(1)
print("After meshgrid: \n", "x-values: \n", X,"\n\n y-values:\n",Y, "\n")
Z = f(X, Y)                 #(2)
    # Notes:
    # (1) Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields over N-D grids,
    # given one-dimensional coordinate arrays x1, x2,…, xn

    # (2) inputs

plt.contourf(X, Y, Z, 100, cmap='Spectral')
plt.colorbar();

plt.show()


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

