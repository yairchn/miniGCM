import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

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

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.Restart_2.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    fig = plt.figure(varname,figsize=(3,6))
    for i in range(n):
        print('i',i)
        print('min max np.mean(var[200:1000,:,i],axis=0) ', np.amin(np.mean(var[200:1000,:,i],axis=0)), np.amax(np.mean(var[200:1000,:,i],axis=0)))
        print('lat_list.shape',lat_list.shape)
        print('var.shape',var.shape)
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.plot(np.mean(var[200:1000,:,i],axis=0),np.array(lat_list),'-k')
        ax1.set_ylabel('latitude / $\circ$')
        ax1.set_xlim(-2,2)
        plt.grid(linestyle=':',alpha=0.6,linewidth=1)
        if i==0: plt.title("Meridional Wind")
        if i==n-1:
            ax1.set_xlabel("$\overline{v}$ / m s$^{-1}$")
        plt.tight_layout()
    plt.savefig('v_3layers_'+runname+'.png')
if __name__ == '__main__':
    main()
