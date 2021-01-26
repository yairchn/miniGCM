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
    runname='410k2'
    #runname='4e5k0'
    runname='310k2'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.Restart_2.nc'
    data = nc.Dataset(ncfile, 'r')

  
    runname2='007e8'
    folder2= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname2+'/stats/'
    ncfile2= folder2+ 'Stats.HeldSuarez.Restart_1.nc'
    print('ncfile2: ',ncfile2)
    data2= nc.Dataset(ncfile2,'r')



    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)
    var2= np.array(data2.groups['zonal_mean'].variables[varname])
    t2= np.divide(data2.groups['zonal_mean'].variables['t'],3600.0*24.0)

    fig = plt.figure(varname,figsize=(3,6))
    for i in range(n):
        print('i',i)
        print('min max np.mean(var[200:1000,:,i],axis=0) ', np.amin(np.mean(var[200:1000,:,i],axis=0)), np.amax(np.mean(var[200:1000,:,i],axis=0)))
        print('lat_list.shape',lat_list.shape)
        print('var.shape',var.shape)
        ax1 = fig.add_subplot(n, 1, i+1)
        im1 = ax1.plot(np.mean(var[200:1000,:,i],axis=0),np.array(lat_list),'-k',label='python')
        im2 = ax1.plot(np.mean(var2[200:800,:,i],axis=0),np.array(lat_list),'-g',label='cython')
        ax1.set_ylabel('latitude / $\circ$')
        ax1.set_xlim(-8,25)
        plt.grid(linestyle=':',alpha=0.6,linewidth=1)
        if i==0: plt.title("Zonal Wind")
        if i==n-1:
            ax1.legend()
            ax1.set_xlim(-6,12)
            ax1.set_xlabel("$\overline{u}$ / m s$^{-1}$")
        plt.tight_layout()
    plt.savefig('u_3layers_'+runname+'.png')
if __name__ == '__main__':
    main()
