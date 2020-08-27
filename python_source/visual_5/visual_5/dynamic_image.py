import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.style.use('dark_background')
fig = plt.figure()

x = np.linspace(0,np.pi*2,1000)
n = 10
L = 1

def triangle_wave(x,n):
    return (pow(-1,(n-1)/2))/(n**2) * np.sin((n*np.pi*x)/L)


#def f(x, y):
#    return np.sin(x) + np.cos(y)

#x = np.linspace(0, 2 * np.pi, 120)
#y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame

y = [0] * 1000
k = 0
ims = []
for m in range(1,n+1): # Is it possible to have m be a decimal value?
    if (m % 2) == 0:
        continue
    y1 = triangle_wave(x,m)
    y[k] = y[k-1] + y1
    im, = plt.plot(x,y[k],animated = True)
    ims.append([im])
    k = k + 1


#for i in range(100):
#    #x += np.pi / 15.
#    #y += np.pi / 20.
#    im = plt.imshow(triangle_wave(x,i), animated=True)
#    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1500, blit=True,
                                repeat_delay=100)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
