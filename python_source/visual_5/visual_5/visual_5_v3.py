import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


plt.style.use('dark_background')
fig = plt.figure()

x = np.linspace(0,np.pi*2,1000)
n = 10
L = 1

def triangle_wave(x,n):
    return (pow(-1,(n-1)/2))/(n**2) * np.sin((n*np.pi*x)/L)

y = [0] * 1000
k = 0
ims = []
for m in range(1,n+1):
    if (m % 2) == 0:
        continue
    y1 = triangle_wave(x,m)
    y[k] = y[k-1] + y1
    im, = plt.plot(x,y[k],animated = True)
    ims.append([im])
    k = k + 1



anim = animation.ArtistAnimation(fig, ims, interval=1500, blit=True,
                                repeat_delay=100)

writer = PillowWriter(fps=1) 
anim.save('visual_5.gif', writer=writer)

plt.show()
