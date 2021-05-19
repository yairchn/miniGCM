import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import shtns
import sphTrans as sph

hres_scaling=1

nlons  = hres_scaling*512  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)   # number of lats for gaussian grid.

# parameters for test
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5  # rotation rate
grav    = 9.80616   # gravity

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lats) # coriolis

latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)


it=25142400
it=34128000
it=15724800
it=504*24*3600
print('day ',it/(24*3600))
it=str(it)

folder='/home/josefs/miniGCM/Output.HeldSuarez.HighResolRun/Fields/'

data=netCDF4.Dataset(folder+'Pressure_'+it+'.nc','r')
ps=data.variables['Pressure'][:,:]/1.e2 # pressure in [hPa]

plt.figure(figsize=(8,4))
plt.clf(); plt.contourf(lonDeg,latDeg,ps[:,:],cmap='bwr',levels=np.linspace(950,1050,35));
plt.colorbar(orientation='vertical',ticks=np.arange(960,1041,20));
plt.ylabel('Latitude / $\circ$')
plt.xlabel('Longitude / $\circ$')
plt.tight_layout()
plt.savefig('ps.png')

print('done')
