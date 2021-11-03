import matplotlib.pyplot as plt
import netCDF4
import numpy as np

it=25142400
it=34128000
it=3024000
it=4579200
it=222*24*3600
it=63590400
it=68947200
print('day ',it/(24*3600))
it=str(it)

lats=np.rad2deg(np.load('/home/scoty/vis/lats.npy'))
lons=np.rad2deg(np.load('/home/scoty/vis/lons.npy'))

data=netCDF4.Dataset('Temperature_'+it+'.nc','r')
T=data.variables['Temperature'][:,:,:]


plt.clf(); plt.contourf(T[:,:,0]); plt.colorbar(); plt.savefig('T0.png')
plt.clf(); plt.contourf(T[:,:,1]); plt.colorbar(); plt.savefig('T1.png')
plt.clf(); plt.contourf(T[:,:,2]); plt.colorbar(); plt.savefig('T2.png')

data=netCDF4.Dataset('V_'+it+'.nc','r')
V=data.variables['V'][:,:,:]

plt.clf(); plt.contourf(V[:,:,0]); plt.colorbar(); plt.savefig('V0.png')
plt.clf(); plt.contourf(V[:,:,1]); plt.colorbar(); plt.savefig('V1.png')
plt.clf(); plt.contourf(V[:,:,2]); plt.colorbar(); plt.savefig('V2.png')

data=netCDF4.Dataset('U_'+it+'.nc','r')
U=data.variables['U'][:,:,:]

plt.clf(); plt.contourf(U[:,:,0]); plt.colorbar(); plt.savefig('U0.png')
plt.clf(); plt.contourf(U[:,:,1]); plt.colorbar(); plt.savefig('U1.png')
plt.clf(); plt.contourf(lons,lats,U[:,:,2],cmap='coolwarm',extend='both',levels=np.linspace(-1.5,2.0,8)); plt.colorbar(orientation='horizontal',extend='both');plt.title('$u_2$'); plt.savefig('U2.png')

data=netCDF4.Dataset('Pressure_'+it+'.nc','r')
ps=data.variables['Pressure'][:,:]

plt.clf(); plt.contourf(ps[:,:]); plt.colorbar(); plt.savefig('ps.png')

print('done')
