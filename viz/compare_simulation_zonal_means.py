import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/compare_simulation_zonal_means.py zonal_mean_U
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    args = parser.parse_args()
    varname = args.varname

    folder1 = os.getcwd() + '/Output.HeldSuarezMoist._PMFS/stats/'
    folder2 = os.getcwd() + '/Output.HeldSuarezMoist.cnon2/stats/'
    ncfile1 = folder1 + 'Stats.HeldSuarezMoist.nc'
    ncfile2 = folder2 + 'Stats.HeldSuarezMoist.nc'
    data1 = nc.Dataset(ncfile1, 'r')
    data2 = nc.Dataset(ncfile2, 'r')

    lat = np.array(data1.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data1.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data1.groups['coordinates'].variables['latitude_list'])
    t = np.divide(data1.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var1 = np.array(data1.groups['zonal_mean'].variables[varname])
    var2 = np.array(data2.groups['zonal_mean'].variables[varname])
    vardiff = np.subtract(var1, var2)
    X, Y = np.meshgrid(t,lat_list)
    fig = plt.figure(varname)
    for i in range(n):
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.contourf(X,Y,np.fliplr(np.rot90(np.squeeze(vardiff[:,:,i]), k=3)))
        ax1.set_ylabel('degree latitude')
        if i<n-1:
            xlabels = [item.get_text() for item in ax1.get_xticklabels()]
            xempty_string_labels = [''] * len(xlabels)
            ax1.set_xticklabels(xempty_string_labels)
        else:
            ax1.set_xlabel('time days')
        fig.colorbar(im1)
    plt.show()
if __name__ == '__main__':
    main()
