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
    varname = 'zonal_mean_P'
    runname='410k2'
    runname='4e5k0'
    runname='310k2'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.Restart_2.nc'
    print('ncfile: ',ncfile)
    data = nc.Dataset(ncfile, 'r')
    runname2='1f3ca'
    folder2= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname2+'/stats/'
    ncfile2= folder2+ 'Stats.HeldSuarez.nc'
    print('ncfile2: ',ncfile2)
    data2= nc.Dataset(ncfile2,'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var2= np.array(data2.groups['zonal_mean'].variables[varname])
    t2= np.divide(data2.groups['zonal_mean'].variables['t'],3600.0*24.0)

    nt2=t2.shape[0]
    print('nt2',nt2)

    ps=var[:,:,2]/100.
    ps2=var2[:,:,2]/100.
    ps_time=np.zeros(1000)
    ps_time2=np.zeros(nt2)
    for it in range(0,1000): ps_time[it]=np.sum(ps[it,:]*weights)
    for it in range(0,nt2): ps_time2[it]=np.sum(ps2[it,:]*weights)

    eps=.1

    plt.figure(figsize=(5,2))
    plt.plot(t,ps_time,'-k')
    plt.plot(t2,ps_time2,'-g')
    plt.xlabel('time / days')
    plt.ylabel('$p_s$ / hPa')
    plt.ylim(1000.-eps,1000.+eps)
    plt.xlim(0,1000.)
    plt.title('Area Mean Surface Pressure')
    plt.tight_layout()
    plt.savefig('timeseries_ps_'+runname+'.png')
if __name__ == '__main__':
    main()
