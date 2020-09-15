import numpy as np
import matplotlib
import sys
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import astropy.io.fits as fits
"""
    # plt.style.use('ggplot')

    # Constants
    i = j = v = 1
    a_ij = i * 1
    L = 5
    


    # Created a figure and configured its size
    fig, ax = plt.subplots()

    # Values we create
    x = np.linspace(0, np.pi, 100)
    t = np.linspace(0, 25, 300)
    y = np.linspace(0, np.pi, 100)

    # Then apply meshgrid onto
    X3, Y3, T3 = np.meshgrid(x, y, t) # each var3 is an array

    kivalue = i * np.pi / L
    kjvalue = j * np.pi / L
    ki_values = [kivalue, kivalue * 3, kivalue * 9]
    kj_values = [kjvalue, kjvalue * 3, kjvalue * 9]
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

    n = 1
    gg = [0]
    # This is the loop I need to make to run through the k values.
    for m in range(0,n+1):
        gg = gg + (g(T3, ki_values, kj_values, m) * f(X3, Y3, ki_values, kj_values, m))
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

"""

ell, DlTT = np.loadtxt("camb_49667121_scalcls.txt", usecols=(0, 1), unpack=True)

## variables to set up the size of the map
N = 2**10  # this is the number of pixels in a linear dimension
            ## since we are using lots of FFTs this should be a factor of 2^N
pix_size  = 0.5 # size of a pixel in arcminutes

## variables to set up the map plots
c_min = -400  # minimum for color bar
c_max = 400   # maximum for color bar
X_width = N*pix_size/60.  # horizontal map width in degrees
Y_width = N*pix_size/60.  # vertical map width in degrees

i = j = v = 1
a_ij = i * 1
L = 5
t = np.linspace(0, 25, 300)
T = np.meshgrid(t)
kivalue = i * np.pi / L
kjvalue = j * np.pi / L
ki_values = [kivalue, kivalue * 3, kivalue * 9]
kj_values = [kjvalue, kjvalue * 3, kjvalue * 9]


def omega(ki, kj, n):
    return np.sqrt(ki[n] ** 2 + kj[n] ** 2) * v

# This is the time dependent portion of the wave equation
def g(t,ki,kj,n):
    return a_ij * np.cos(omega(ki,kj,n) * t)

def make_CMB_T_map(N,pix_size,ell,DlTT):
    global T, ki_values, kj_values, m
    "makes a realization of a simulated CMB sky map given an input DlTT as a function of ell," 
    "the pixel size (pix_size) required and the number N of pixels in the linear dimension."
    #np.random.seed(100)
    # convert Dl to Cl
    ClTT = DlTT * 2 * np.pi / (ell*(ell+1.))
    ClTT[0] = 0. # set the monopole and the dipole of the Cl spectrum to zero
    ClTT[1] = 0.

    # make a 2D real space coordinate system
    onesvec = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.) # create an array of size N between -0.5 and +0.5
    # compute the outer product matrix: X[i, j] = onesvec[i] * inds[j] for i,j 
    # in range(N), which is just N rows copies of inds - for the x dimension
    X = np.outer(onesvec,inds) 
    # compute the transpose for the y dimension
    Y = np.transpose(X)
    # radial component R
    R = np.sqrt(X**2. + Y**2.)
    
    # now make a 2D CMB power spectrum
    pix_to_rad = (pix_size/60. * np.pi/180.) # going from pix_size in arcmins to degrees and then degrees to radians
    ell_scale_factor = 2. * np.pi /pix_to_rad  # now relating the angular size in radians to multipoles
    ell2d = R * ell_scale_factor # making a fourier space analogue to the real space R vector
    ClTT_expanded = np.zeros(int(ell2d.max())+1) 
    # making an expanded Cl spectrum (of zeros) that goes all the way to the size of the 2D ell vector
    ClTT_expanded[0:(ClTT.size)] = ClTT # fill in the Cls until the max of the ClTT vector

    # the 2D Cl spectrum is defined on the multiple vector set by the pixel scale
    CLTT2d = ClTT_expanded[ell2d.astype(int)] 
    #plt.imshow(np.log(CLTT2d))
        
    
    # now make a realization of the CMB with the given power spectrum in real space
    random_array_for_T = np.random.normal(0,1,(N,N))
    FT_random_array_for_T = np.fft.fft2(random_array_for_T)   # take FFT since we are in Fourier space 
    
    FT_2d = np.sqrt(CLTT2d) * FT_random_array_for_T # we take the sqrt since the power spectrum is T^2
    plt.imshow(np.real(FT_2d))
        
    
    ## make a plot of the 2D cmb simulated map in Fourier space, note the x and y axis labels need to be fixed
    #Plot_CMB_Map(np.real(np.conj(FT_2d)*FT_2d*ell2d * (ell2d+1)/2/np.pi),0,np.max(np.conj(FT_2d)*FT_2d*ell2d * (ell2d+1)/2/np.pi),ell2d.max(),ell2d.max())  ###
    
    # move back from ell space to real space
    CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
    # move back to pixel space for the map
    CMB_T = CMB_T/(pix_size /60.* np.pi/180.)
    # we only want to plot the real component
    CMB_T = int(np.real(CMB_T)) * int(g(T, ki_values, kj_values, m))

    ## return the map
    return(CMB_T)
  ###############################

def Plot_CMB_Map(Map_to_Plot,c_min,c_max,X_width,Y_width):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    print("map mean:",np.mean(Map_to_Plot),"map rms:",np.std(Map_to_Plot))
    plt.gcf().set_size_inches(10, 10)
    im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
    im.set_clim(c_min,c_max)
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    #cbar = plt.colorbar()
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel('angle $[^\circ]$')
    plt.xlabel('angle $[^\circ]$')
    cbar.set_label('tempearture [uK]', rotation=270)
    
    plt.show()
    return(0)
  ###############################
m = 1
## make a CMB T map
CMB_T = make_CMB_T_map(N,pix_size,ell,DlTT)
Plot_CMB_Map(CMB_T,c_min,c_max,X_width,Y_width)
plt.savefig('cmb1.png')
plt.clf()