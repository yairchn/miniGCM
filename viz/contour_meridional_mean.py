import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/contour_meridional_mean.py meridional_mean_V
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    args = parser.parse_args()
    varname = args.varname

    folder = os.getcwd() + '/Output.HeldSuarez_moist.20day/stats/'
    ncfile = folder + 'Stats.HeldSuarez_moist.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['longitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['longitude_list'])
    var = np.array(data.groups['meridional_mean'].variables[varname])
    t = np.divide(data.groups['meridional_mean'].variables['t'],3600.0*24.0)

    X, Y = np.meshgrid(t,lat_list)
    print(n)
    fig = plt.figure(varname)
    for i in range(n):
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.contourf(X,Y,np.fliplr(np.rot90(np.squeeze(var[:,:,i]), k=3)))
        ax1.set_ylabel('degree longitude')
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
