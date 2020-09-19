import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

plt.style.use('dark_background')
fig = plt.figure()
plt.axis(xmin = 0, xmax = 2)

x = np.linspace(0,np.pi*2,1000)
n = 20
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

"""
    I think what needs to be done now is control the amount of frames produced.
    As of now it's only doing a set amount that zips by really fast. 
"""


anim = animation.ArtistAnimation(fig, ims, interval=15, blit=True)
"""
    anim = animation.ArtistAnimation(fig, ims, interval=1500, blit=True,
                                    repeat_delay=100)
"""

anim.save('visual_5.mp4',writer = writer,fps = 120)

plt.show()
