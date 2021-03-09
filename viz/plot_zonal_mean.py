import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/plot_zonal_mean.py zonal_mean_U 1800.0
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    parser.add_argument("mytime",  type=float)
    args = parser.parse_args()
    varname = args.varname
    mytime = args.mytime

    folder = os.getcwd() + '/Output.HeldSuarez.dlayer_moist/stats/'
    ncfile = folder + 'Stats.HeldSuarez.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    timescale = 3600.0*24.0
    t = np.divide(data.groups['zonal_mean'].variables['t'],timescale)

    # t0 = numpy.where(numpy.diff(numpy.signbit(t-mytime)))[0]
    t0 = np.argmin(np.abs(np.subtract(t,mytime-0.1)))
    t1 = np.argmin(np.abs(np.subtract(t,mytime+0.1)))
    t0 = 200
    t1 = 260
    fig = plt.figure(varname)
    fig.add_subplot(3, 1, 1)
    plt.plot(lat_list,var[0,:,0],0)
    plt.ylim([285,295])
    fig.add_subplot(3, 1, 2)
    plt.plot(lat_list,var[0,:,1],0)
    plt.ylim([290,300])
    fig.add_subplot(3, 1, 3)
    plt.plot(lat_list,var[0,:,2],0)
    plt.ylim([293,308])
    # for i in range(n):
    #     ax1 = fig.add_subplot(n, 1, i+1)
    #     im1 = ax1.plot(lat_list,var[-1,:,i],0)
    #     im1 = ax1.set_ylim([280,295])
    #     # ax1.set_ylabel(varname)
    #     # if i<n-1:
    #     #     xlabels = [item.get_text() for item in ax1.get_xticklabels()]
    #     #     xempty_string_labels = [''] * len(xlabels)
    #     #     ax1.set_xticklabels(xempty_string_labels)
    #     # else:
    #     #     ax1.set_xlabel('degree latitude')
    plt.show()
if __name__ == '__main__':
    main()
