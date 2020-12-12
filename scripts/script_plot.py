import numpy as np
import matplotlib.pyplot as plt
import netCDF4

data=netCDF4.Dataset('T_8640000.nc','r')
T=data.variables['T'][:,:,:]

data=netCDF4.Dataset('u_8640000.nc','r')
u=data.variables['u'][:,:,:]

abs_zero=-273.5 # °C

print('T layer 1 min max [°C]',T[:,:,0].min()+abs_zero,T[:,:,0].max()+abs_zero)
print('T layer 2 min max [°C]',T[:,:,1].min()+abs_zero,T[:,:,1].max()+abs_zero)
print('T layer 3 min max [°C]',T[:,:,2].min()+abs_zero,T[:,:,2].max()+abs_zero)


print('u layer 1 min max [m/s]',u[:,:,0].min(),u[:,:,0].max())
print('u layer 2 min max [m/s]',u[:,:,1].min(),u[:,:,1].max())
print('u layer 3 min max [m/s]',u[:,:,2].min(),u[:,:,2].max())


plt.contourf(T[:,:,0],levels=np.linspace(T[:,:,0].min(),T[:,:,0].max(),40),cmap='coolwarm'); plt.colorbar(); plt.savefig('T1.png'); plt.clf()
plt.contourf(T[:,:,1],levels=np.linspace(T[:,:,1].min(),T[:,:,1].max(),40),cmap='coolwarm'); plt.colorbar(); plt.savefig('T2.png'); plt.clf()
plt.contourf(T[:,:,2],levels=np.linspace(T[:,:,2].min(),T[:,:,2].max(),40),cmap='coolwarm'); plt.colorbar(); plt.savefig('T3.png'); plt.clf()

plt.contourf(u[:,:,0],levels=np.linspace(u[:,:,0].min(),u[:,:,0].max(),40),cmap='viridis'); plt.colorbar(); plt.savefig('u1.png'); plt.clf()
plt.contourf(u[:,:,1],levels=np.linspace(u[:,:,1].min(),u[:,:,1].max(),40),cmap='viridis'); plt.colorbar(); plt.savefig('u2.png'); plt.clf()
plt.contourf(u[:,:,2],levels=np.linspace(u[:,:,2].min(),u[:,:,2].max(),40),cmap='viridis'); plt.colorbar(); plt.savefig('u3.png'); plt.clf()
