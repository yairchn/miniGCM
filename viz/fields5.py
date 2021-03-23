import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import os

folder = os.getcwd() + '/Output.HeldSuarez.theta6_efold/Fields/'
it=56246400
# it=42336000
print('day ',it/(24*3600))
it=str(it)

data=netCDF4.Dataset(folder+'Temperature_'+it+'.nc','r')
T=data.variables['Temperature'][:,:,:]


plt.clf(); plt.contourf(T[:,:,0]); plt.colorbar(); plt.savefig('T0.png')
plt.clf(); plt.contourf(T[:,:,1]); plt.colorbar(); plt.savefig('T1.png')
plt.clf(); plt.contourf(T[:,:,2]); plt.colorbar(); plt.savefig('T2.png')
# plt.clf(); plt.contourf(T[:,:,3]); plt.colorbar(); plt.savefig('T3.png')
# plt.clf(); plt.contourf(T[:,:,4]); plt.colorbar(); plt.savefig('T4.png')

data=netCDF4.Dataset(folder+'V_'+it+'.nc','r')
V=data.variables['V'][:,:,:]

plt.clf(); plt.contourf(V[:,:,0]); plt.colorbar(); plt.savefig('V0.png')
plt.clf(); plt.contourf(V[:,:,1]); plt.colorbar(); plt.savefig('V1.png')
plt.clf(); plt.contourf(V[:,:,2]); plt.colorbar(); plt.savefig('V2.png')
# plt.clf(); plt.contourf(V[:,:,3]); plt.colorbar(); plt.savefig('V3.png')
# plt.clf(); plt.contourf(V[:,:,4]); plt.colorbar(); plt.savefig('V4.png')

data=netCDF4.Dataset(folder+'U_'+it+'.nc','r')
U=data.variables['U'][:,:,:]

plt.clf(); plt.contourf(U[:,:,0]); plt.colorbar(); plt.savefig('U0.png')
plt.clf(); plt.contourf(U[:,:,1]); plt.colorbar(); plt.savefig('U1.png')
plt.clf(); plt.contourf(U[:,:,2]); plt.colorbar(); plt.savefig('U2.png')
# plt.clf(); plt.contourf(U[:,:,3]); plt.colorbar(); plt.savefig('U3.png')
# plt.clf(); plt.contourf(U[:,:,4]); plt.colorbar(); plt.savefig('U4.png')

data=netCDF4.Dataset(folder+'Pressure_'+it+'.nc','r')
ps=data.variables['Pressure'][:,:]

plt.clf(); plt.contourf(ps[:,:]); plt.colorbar(); plt.savefig('ps.png')

print('done')
