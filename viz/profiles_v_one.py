import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os
from scipy.signal import savgol_filter

# command line:
# python viz/contour_zonal_mean.py zonal_mean_P
def main():
    #parser = argparse.ArgumentParser(prog='miniGCM')
    #parser.add_argument("varname")
    #args = parser.parse_args()
    #varname = args.varname
    varname = 'zonal_mean_V'
    runname='410k2'
    #runname='4e5k0'
    runname='310k2'
    runname='43809'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    print('var.shape', var.shape) #steps of 5 days

    fig= plt.figure(figsize=(4,4))
    im1 = plt.plot(savgol_filter(np.mean(var[50:60,:,0],axis=0),5,1),np.array(lat_list),'-k',linewidth=4,alpha=0.4,label='top')
    im2 = plt.plot(savgol_filter(np.mean(var[50:60,:,1],axis=0),5,1),np.array(lat_list),'-k',linewidth=2,alpha=0.6,label='middle')
    im3 = plt.plot(savgol_filter(np.mean(var[50:60,:,2],axis=0),5,1),np.array(lat_list),'-k',linewidth=1,alpha=0.8,label='bottom')
    plt.ylabel('Latitude / $\circ$')
    plt.grid(linestyle=':',alpha=0.6,linewidth=1)
    plt.title("Meridional Wind")
    #plt.xlim(-6,26)
    plt.xlabel("$\overline{v}$ / m s$^{-1}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig('v_3layers_'+runname+'.png',dpi=150)
if __name__ == '__main__':
    main()
