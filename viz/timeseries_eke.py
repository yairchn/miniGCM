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
    varname = 'global_mean_KE'
    runname='.44-test3lay'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.nc'
    print('ncfile: ',ncfile)
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['global_mean'].variables[varname])
    t = np.divide(data.groups['global_mean'].variables['t'],3600.0*24.0)

    print('global mean energy shape',var.shape)

    eke=var

    plt.figure(figsize=(5,5))
    plt.plot(t,eke[:,2],'-k',linewidth=1,alpha=0.6,label='bottom')
    plt.plot(t,eke[:,1],'-k',linewidth=1,alpha=0.8,label='middle')
    plt.plot(t,eke[:,0],'-k',linewidth=1,alpha=1.0,label='top')
    plt.xlabel('time / days')
    plt.ylabel('$ke$ / m$^2$ s$^2$')
    #plt.ylim(1000.-eps,1000.+eps)
    plt.xlim(0,1000.)
    plt.title('Layer Mean Kinetic Energy')
    plt.legend(ncol=3,loc='upper right')
    plt.tight_layout()
    plt.savefig('timeseries_ke_'+runname+'.pdf')
if __name__ == '__main__':
    main()
