import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
# plt.style.use('ggplot')

# Constants
i = j = v = 1
a_ij = i * 1
L = 5
k_i = i * np.pi / L 
k_j = j * np.pi / L
k_i2 = 3 * k_i 
k_j2 = 3 * k_j 
k_i3 = 3 * k_i2 
k_j3 = 3 * k_j2 
w_ij = np.sqrt(k_i ** 2 + k_j ** 2) * v

# Created a figure and configured its size
fig, ax = plt.subplots()

# Values we create
x = np.linspace(0, np.pi, 100)
t = np.linspace(0, 25, 300)
y = np.linspace(0, np.pi, 100)

# Then apply meshgrid onto
X3, Y3, T3 = np.meshgrid(x, y, t) # each var3 is an array

# our Equation: a_ij * np.cos(w_ij * t) * np.sin(k_i * x) * np.sin(k_j * y)

t_dependent = a_ij * np.cos(w_ij*T3)
G = t_dependent * np.sin(k_i * X3) * np.sin(k_j * Y3) + \
    t_dependent * np.sin(k_i2 * X3) * np.sin(k_j2 * Y3) + t_dependent * np.sin(k_i3 * X3) * np.sin(k_j3 * Y3)

print("This is the data we want to get:\n\n",G[:-1, :-1, 0],"\n\n=====================================================")


#cax = ax.pcolormesh(x, y, G[:-1, :-1, 0], vmin=-1, vmax=1, cmap='Spectral')
# G[:-1, :-1, 0] indicates all the 



#fig.colorbar(cax)
 
#def animate(i):
#     cax.set_array(G[:-1, :-1, i].flatten())


#anim = animation.FuncAnimation(fig, animate, interval=150, frames=len(t)-1)

#writer = PillowWriter(fps=25) 
##anim.save('visual_1.gif', writer=writer)

#plt.draw()
#plt.show()