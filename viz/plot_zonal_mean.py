import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/plot_zonal_mean.py zonal_mean_qv_star
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    args = parser.parse_args()
    varname = args.varname

    folder = os.getcwd() + '/Output.HeldSuarezMoist.vstar_mss005/stats/'
    ncfile = folder + 'Stats.HeldSuarezMoist.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    X, Y = np.meshgrid(t,lat_list)
    fig = plt.figure(varname)
    for i in range(n):
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.plot(np.mean(var[80:-1,:,i], axis = 0), lat_list)
        ax1.set_ylabel('degree latitude')
        if i==n-1:
            ax1.set_xlabel(varname)
        # fig.colorbar(im1)
    fig = plt.figure('one plot')
    plt.ylabel('degree latitude')
    for i in range(n):
        plt.plot(np.mean(var[80:-1,:,i], axis = 0), lat_list,  label=str(i))
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
