import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

from math import *

weights=np.cos(2.*pi*np.linspace(-90.,90.,64)/360.)
weights/=np.sum(weights)
print('weights.shape and sum', weights.shape, np.sum(weights))


# command line:
# python viz/contour_zonal_mean.py zonal_mean_P
def main():
    #parser = argparse.ArgumentParser(prog='miniGCM')
    #parser.add_argument("varname")
    #args = parser.parse_args()
    #varname = args.varname
    varname = 'zonal_mean_Ps'
    runname='410k2'
    runname='4e5k0'
    runname='310k2'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = os.getcwd() + '/Output.HeldSuarez.with_pgf_cor/stats/Stats.HeldSuarez.nc'
    print('ncfile: ',ncfile)
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = 3

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['surface_zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    ps=var[:,:]/100.
    ps_time=np.zeros(1000)
    for it in range(0,1000): ps_time[it]=np.sum(ps[it,:]*weights)

    eps=.1

    plt.figure(figsize=(5,2))
    plt.plot(t,ps_time,'-k')
    plt.xlabel('time / days')
    plt.ylabel('$p_s$ / hPa')
    # plt.ylim(1000.-eps,1000.+eps)
    plt.xlim(0,1000.)
    plt.title('Area Mean Surface Pressure')
    plt.tight_layout()
    plt.show()
    plt.savefig('timeseries_ps_'+runname+'.png')
if __name__ == '__main__':
    main()
