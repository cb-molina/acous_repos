import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
# plt.style.use('ggplot')

# Constants
i = j = v = 1
a_ij = i * 1
L = 5
kivalue = i * np.pi / L
kjvalue = j * np.pi / L
ki_values = [kivalue, kivalue * 3, kivalue * 9]
kj_values = [kjvalue, kjvalue * 3, kjvalue * 9]


# Created a figure and configured its size
fig, ax = plt.subplots()

# Values we create
x = np.linspace(0, np.pi, 100)
t = np.linspace(0, 25, 300)
y = np.linspace(0, np.pi, 100)

# Then apply meshgrid onto
X3, Y3, T3 = np.meshgrid(x, y, t) # each var3 is an array

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
    gg = gg + (g(T3, ki_values, kj_values, m) * f(X3, Y3, ki_values, kj_values, m))

cax = ax.pcolormesh(x, y, gg[:-1, :-1, 0], vmin=-1, vmax=1, cmap='Spectral')
fig.colorbar(cax) 
def animate(i):
     cax.set_array(gg[:-1, :-1, i].flatten())

anim = animation.FuncAnimation(fig, animate, interval=150, frames=len(t)-1)

writer = PillowWriter(fps=25) 
#anim.save('visual_1_v2.gif', writer=writer)

plt.draw()
plt.show()