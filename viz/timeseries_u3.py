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
    #parser = argparse.ArgumentParser(prog='miniGCM')
    #parser.add_argument("varname")
    #args = parser.parse_args()
    #varname = args.varname
    varname = 'zonal_mean_U'
    runname='410k2'
    runname='4e5k0'
    runname='310k2'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.Restart_2.nc'
    print('ncfile: ',ncfile)
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    u3=var[:,:,2]
    u3_weights=np.copy(u3)*0.
    u3_time=np.zeros(1000)
    for it in range(0,1000): u3_weights[it,:]=u3[it,:]*weights
    for it in range(0,1000): u3_time[it]=np.sum(u3_weights[it,:])

    print('min max u3     ',np.amin(u3     ),np.amax(u3     ))
    print('min max u3_weights',np.amin(u3_weights),np.amax(u3_weights))
    print('min max u3_time',np.amin(u3_time),np.amax(u3_time))

    eps=1.

    plt.figure(figsize=(5,2))
    plt.plot(t,u3_time,'-k')
    plt.xlabel('time / days')
    plt.ylabel('$u_3$ / m s$^{-1}$')
    plt.ylim(0.-eps,0.+eps)
    plt.xlim(0,1000.)
    plt.title('Area Mean Surface Layer Wind')
    plt.tight_layout()
    plt.savefig('timeseries_u3_'+runname+'.png')
if __name__ == '__main__':
    main()
