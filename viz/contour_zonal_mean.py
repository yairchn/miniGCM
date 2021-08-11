import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/contour_zonal_mean.py zonal_mean_QT
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    args = parser.parse_args()
    varname = args.varname

    folder = os.getcwd() + '/Output.HeldSuarezMoist.o_truncation/stats/'
    ncfile = folder + 'Stats.HeldSuarezMoist.Rerun_3.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    X, Y = np.meshgrid(t,lat_list)
    print(np.shape(var))
    print(np.shape(X))
    fig = plt.figure(varname)
    for i in range(n):
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.contourf(X,Y,np.fliplr(np.rot90(np.squeeze(var[:,:,i]), k=3)))
        ax1.set_ylabel('lat')
        if i<n-1:
            xlabels = [item.get_text() for item in ax1.get_xticklabels()]
            xempty_string_labels = [''] * len(xlabels)
            ax1.set_xticklabels(xempty_string_labels)
        else:
            ax1.set_xlabel('time days')
        fig.colorbar(im1)
    globle_mean_var = np.mean(np.mean(var, axis = 1), axis = 1)
    norm_var  = np.divide(np.subtract(globle_mean_var,globle_mean_var[0]),globle_mean_var[0])
    pressure_levels = np.flipud(np.array([250.0, 350.0, 450.0, 550.0, 650.0, 750.0, 850.0, 950.0]))
    y,z = np.meshgrid(lat_list,pressure_levels)
    plt.figure('global mean')
    plt.plot(t, norm_var)
    print(np.shape(np.fliplr(np.rot90(np.squeeze(np.mean(var[700:799,:,0:],axis = 0))))))
    print(np.shape(y))
    print(np.shape(z))
    plt.figure('time mean ' + varname)
    plt.contourf(y,z, np.rot90(np.squeeze(np.mean(var[700:799,:,0:],axis = 0))))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('degree latitude')
    plt.show()

if __name__ == '__main__':
    main()
