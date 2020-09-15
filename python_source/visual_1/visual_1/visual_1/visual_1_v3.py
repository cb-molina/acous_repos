import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
# plt.style.use('ggplot')

# Constants
i = j = v = 1
a_ij = i * 1
L = 5

"""
    k_i = i * np.pi / L 
    k_j = j * np.pi / L
    k_i2 = 3 * k_i 
    k_j2 = 3 * k_j 
    k_i3 = 3 * k_i2 
    k_j3 = 3 * k_j2
    k_i4 = 3 * k_i3 
    k_j4 = 3 * k_j3
    k_i5 = 3 * k_i4 
    k_j5 = 3 * k_j4
    w_ij = np.sqrt(k_i ** 2 + k_j ** 2) * v
"""


# Created a figure and configured its size
fig, ax = plt.subplots()

# Values we create
x = np.linspace(0, np.pi, 100)
t = np.linspace(0, 25, 300)
y = np.linspace(0, np.pi, 100)

# Then apply meshgrid onto
X3, Y3, T3 = np.meshgrid(x, y, t) # each var3 is an array

# our Equation: a_ij * np.cos(w_ij * t) * np.sin(k_i * x) * np.sin(k_j * y)
"""
    t_dependent = a_ij * np.cos(w_ij*T3)
    G = t_dependent * np.sin(k_i * X3) * np.sin(k_j * Y3) + \
        t_dependent * np.sin(k_i2 * X3) * np.sin(k_j2 * Y3) + t_dependent * np.sin(k_i3 * X3) * np.sin(k_j3 * Y3)

    print("This is the data we want to get:\n\n",G[:-1, :-1, 0],"\n\n=====================================================")
"""

kivalue = i * np.pi / L
kjvalue = j * np.pi / L

def ki_calc(n):
    global kjvalue
    ki_values = [kivalue]
    if n == 0:
        return ki_values
    for i in range(n):
        kival = ki_values[i] * 3
        ki_values.append(kival)
    return ki_values

def kj_calc(n):
    global kjvalue
    kj_values = [kjvalue]
    if n == 0:
        return kj_values
    for i in range(n):
        kjval = kj_values[i] * 3
        kj_values.append(kjval)
    return kj_values

# indexing of the ki/j_values will be done with n

# We'll make v constant for now
def omega(ki, kj, n):
    return np.sqrt(ki[n] ** 2 + kj[n] ** 2) * v
# This is the time dependent portion of the wave equation
def g(t,ki,kj,n):
    return a_ij * np.cos(omega(ki,kj,n) * t)
# This is the spatial dependent portion of the wave equation
def f(x,y,ki,kj,n):
    return np.sin(ki[n] * x) * np.sin(kj[n] * y)

n = 2
gg = [0]
# This is the loop I need to make to run through the k values.
for m in range(0,n+1):
    gg = gg + (g(T3, ki_calc(n), kj_calc(n), m) * f(X3, Y3, ki_calc(n), kj_calc(n), m))
    print("This is what we have for our new value:\n\n","m is: ",m,"\n\n",gg[:-1, :-1, 0],"\n\n")



# Uncommment this section once you produce the same data as G
# ========================================================================================================

cax = ax.pcolormesh(x, y, gg[:-1, :-1, 0], vmin=-1, vmax=1, cmap='Spectral')
fig.colorbar(cax) 
def animate(i):
     cax.set_array(gg[:-1, :-1, i].flatten())
anim = animation.FuncAnimation(fig, animate, interval=150, frames=len(t)-1)

writer = PillowWriter(fps=140) 
#anim.save('visual_1_v2.gif', writer=writer)

plt.draw()
plt.show()