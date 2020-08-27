import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal

plt.style.use('dark_background')

fig = plt.figure(figsize=(12,6))
fig.set_dpi(100)

x = np.linspace(0,np.pi*2,1000)

n = 50
L = 1

def triangle_wave(x,n):
    return (pow(-1,(n-1)/2))/(n**2) * np.sin((n*np.pi*x)/L)

y = [0] * 1000
k = 0
for m in range(1,n+1): # Is it possible to have m be a decimal value?
    if (m % 2) == 0:
        continue
    y1 = triangle_wave(x,m)
    y[k] = y[k-1] + y1
    plt.plot(x,y[k],color='red')
    k = k + 1




#plt.plot(x,y,color='yellow')
plt.show()