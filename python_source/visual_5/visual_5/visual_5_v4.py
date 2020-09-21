"""
    Christian B Molina
    UC Davis 2020
    
    Version Update: 
    Currently, this version is capable of producing MP4's of the time evolution
    of changing triangle waves

"""

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

# This is our triangle wave equation
def triangle_wave(x,n):
    return (pow(-1,(n-1)/2))/(n**2) * np.sin((n*np.pi*x)/L)

def populateImages(x,n,fL):
    y = [0] * 1000
    k = 0
    images = []
    for m in range(1,n+1):
        if (m % 2) == 0:
            continue
        y1 = triangle_wave(x,m)
        y[k] = y[k-1] + y1
        im, = plt.plot(x,y[k],animated = True)
        images.append([im])
        for j in range(fL):
            images.append([im])
        k = k + 1
    return images

frameLength = 10
ims = populateImages(x,n,frameLength)

anim = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
"""
    anim = animation.ArtistAnimation(fig, ims, interval=1500, blit=True,
                                    repeat_delay=100)
"""

anim.save('visual_5_2.mp4',writer = writer)

plt.show()
