import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

#Set up figure
fig = plt.figure(figsize=(1,1),tight_layout = {'pad': 0})
inclination = 60 #degrees from pole
plotcrs = ccrs.Orthographic(0, 90 - inclination)
ax = plt.subplot(projection=plotcrs)

#Plot, limiting colors to extreme data values
vlim = np.max(np.abs(drm))
ax.pcolormesh(lon*180/np.pi,lat*180/np.pi,drm,transform=ccrs.PlateCarree(),
                        cmap='seismic',vmin=-vlim,vmax=vlim)

#Necessary function calls
ax.relim()
ax.autoscale_view()

#Save the image
plt.savefig('l4m3.png',dpi=300)

"""
    http://keatonb.github.io/archivers/shanimate
"""