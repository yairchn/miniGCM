import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

from math import *

weights=np.cos(2.*pi*np.linspace(-90.,90.,256)/360.)
weights/=np.sum(weights)
print('weights.shape and sum', weights.shape, np.sum(weights))


# command line:
# python viz/contour_zonal_mean.py zonal_mean_P
def main():
    runname='410k2'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/Fields/'
    ncfile = folder + 'u_merge.nc'
    print('ncfile: ',ncfile)
    data = nc.Dataset(ncfile, 'r')
    ncfile = folder + 'v_merge.nc'
    print('ncfile: ',ncfile)
    data2= nc.Dataset(ncfile, 'r')

    u3 = data.variables['u'][:,:,:,2]
    v3 = data2.variables['v'][:,:,:,2]
    u2 = data.variables['u'][:,:,:,1]
    v2 = data2.variables['v'][:,:,:,1]
    u1 = data.variables['u'][:,:,:,0]
    v1 = data2.variables['v'][:,:,:,0]

    t=np.arange(0,1000) #time in days

    print('u1.shape', u1.shape)
    print('u2.shape', u2.shape)
    print('u3.shape', u3.shape)
    print('v1.shape', v1.shape)
    print('v2.shape', v2.shape)
    print('v3.shape', v3.shape)

    ke3_time=np.zeros(1000)
    ke2_time=np.zeros(1000)
    ke1_time=np.zeros(1000)
    for it in range(0,1000):
        ke3_time[it]=np.sum(0.5*np.mean(u3[it,:,:]**2+v3[it,:,:]**2,axis=1)*weights)
        ke2_time[it]=np.sum(0.5*np.mean(u2[it,:,:]**2+v2[it,:,:]**2,axis=1)*weights)
        ke1_time[it]=np.sum(0.5*np.mean(u1[it,:,:]**2+v1[it,:,:]**2,axis=1)*weights)

    plt.figure(figsize=(5.,3.5))
    plt.plot(t,ke1_time,'-k',linewidth=4,alpha=0.4,label='top')
    plt.plot(t[200:1000],t[200:1000]*0.+np.mean(ke1_time[200:1000]),'--r',alpha=0.8,linewidth=0.5)
    plt.plot(t,ke2_time,'-k',linewidth=2,alpha=0.6,label='middle')
    plt.plot(t[200:1000],t[200:1000]*0.+np.mean(ke2_time[200:1000]),'--r',alpha=0.8,linewidth=0.5)
    plt.plot(t,ke3_time,'-k',linewidth=1,alpha=0.8,label='bottom')
    plt.plot(t[200:1000],t[200:1000]*0.+np.mean(ke3_time[200:1000]),'--r',alpha=0.8,linewidth=0.5)
    plt.xlabel('time / days')
    plt.ylabel('KE / m s$^{-1}$')
    plt.xlim(0,1000.)
    plt.legend()
    plt.tight_layout()
    plt.savefig('timeseries_eke_'+runname+'.png',dpi=150)
if __name__ == '__main__':
    main()
