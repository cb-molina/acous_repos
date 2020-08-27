import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(5, 3))
#This sets up the figure and its axis,
ax.set(xlim=(-3, 3), ylim=(-1, 1))
#This fixes the axis limits

#x = np.linspace(-3, 3, 91)
x = np.linspace(0,np.pi*2,1000)

#t = np.linspace(1, 25, 30)
X2 = np.meshgrid(x)

n = 7
L = 1

def triangle_wave(x,n):
    n = float(n)
    return (pow(-1.,(n-1.)/2.))/(n**2.) * np.sin((n*np.pi*x)/L)


#sinT2 = np.sin(2 * np.pi * T2 / T2.max())
#F = 0.9 * sinT2 * np.sinc(X2 * (1 + T2))
#F is a 2D array comprising some arbitrary data to be animated

y = [0] * 1000
k = 0
for m in range(1,n+1): # Is it possible to have m be a decimal value?
    if (m % 2) == 0:
        continue
    y1 = triangle_wave(X2,m)
    y[k] = y[k-1] + y1
    #plt.plot(x,y[k],color='red')
    k = k + 1


line = ax.plot(x, y[0, :], color = 'k', lw = 2)[0]
#This sets up a line object with the desired attributes, whic in this case
#are that it's coloured black and has a line weight of 2.
#Note the [0] at the end, this is necessary becayse the plot command returns a
#list of line objects. Here we are only plotting a single line, 
#so we simply want the first object in the list of lines.

def animate(i):
    line.set_ydata(y[i,:])
#This function needs one argument. The only command in the function
#is to change the object's y data

anim = FuncAnimation(
    fig, animate, interval=100, frames=len(t)-1)
 
plt.draw()
plt.show()