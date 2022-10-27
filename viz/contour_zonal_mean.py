import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line from main folder:
# python viz/contour_zonal_mean.py zonal_mean_U
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    args = parser.parse_args()
    varname = args.varname

    ncfile = os.getcwd() + '/Output.HeldSuarez.a-steps_test/stats/Stats.HeldSuarez.Rerun_8.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    pressure_levels = np.array(data.groups["coordinates"].variables["pressure_levels"])
    n = np.size(pressure_levels) - 1
    pressure_layers = np.flipud((pressure_levels[0:-1]+pressure_levels[1:])/2)

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.multiply(np.array(data.groups['zonal_mean'].variables[varname]) ,1.0)
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    X, Y = np.meshgrid(t,lat_list)
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

    colors = ['r','b','g','r','b','g']
    fig = plt.figure(varname + "zonal mean")
    for i in range(n):
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.plot(np.mean(var[400:-1,:,i],axis = 0),Y, colors[i])
        ax1.set_ylabel('degree latitude')
        if i<n-1:
            xlabels = [item.get_text() for item in ax1.get_xticklabels()]
            xempty_string_labels = [''] * len(xlabels)
            # ax1.set_xticklabels(xempty_string_labels)
        else:
            ax1.set_xlabel('time days')

    fig = plt.figure('unified plot')
    plt.ylabel('degree latitude')
    for i in range(n):
        plt.plot(np.mean(var[400:-1,:,i], axis = 0), lat_list,  label=str(i))
    plt.legend()

    X, Y = np.meshgrid(lat_list,pressure_layers)
    fig = plt.figure('pressure-latitude contours')
    plt.contourf(X,Y,np.rot90(np.mean(var, axis = 0), k=3))
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
