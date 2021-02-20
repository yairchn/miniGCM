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
    varname = 'zonal_mean_U'
    runname='43a10'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])

    fig = plt.figure(varname,figsize=(3,6))
    for i in range(n):
        print('i',i)
        print('lat_list.shape',lat_list.shape)
        print('var.shape',var.shape)
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.plot(np.mean(var[350:400,:,i],axis=0),np.array(lat_list),'-k',label='run1')
        ax1.set_ylabel('latitude / $\circ$')
        ax1.set_xlim(-8,25)
        plt.grid(linestyle=':',alpha=0.6,linewidth=1)
        if i==0:
            plt.title("Zonal Wind")
            ax1.set_xlim(-8,40)
        if i==n-1:
            ax1.legend()
            ax1.set_xlim(-6,12)
            ax1.set_xlabel("$\overline{u}$ / m s$^{-1}$")
        plt.tight_layout()
    plt.savefig('u_3layers_'+runname+'.png')
if __name__ == '__main__':
    main()
