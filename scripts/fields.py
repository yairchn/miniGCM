import matplotlib.pyplot as plt
import netCDF4
import numpy as np

it=str(677*24*3600)

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
plt.clf(); plt.contourf(U[:,:,2]); plt.colorbar(); plt.savefig('U2.png')

data=netCDF4.Dataset('Pressure_'+it+'.nc','r')
ps=data.variables['Pressure'][:,:]

plt.clf(); plt.contourf(ps[:,:]); plt.colorbar(); plt.savefig('ps.png')

print('done')
