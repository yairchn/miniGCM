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
    varname = 'V'
    runname='43a12'
    step=3600*24

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/Fields/'
    ncfile = folder + '../stats/Stats.HeldSuarez.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    lon = np.array(data.groups['coordinates'].variables['longitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    nt=1000
    nt=200
    v_abs=[20,10,5]
    #for it in np.arange(5,nt,2):
    for it in [50,100,150]:
        print('it =',it)
        ncfile = folder + varname+'_'+str(step*it)+'.nc'
        data = nc.Dataset(ncfile, 'r')
        var = np.array(data.variables[varname][:,:,:])
        X, Y = (lon,lat)
        fig = plt.figure(varname,figsize=(6,10))
        for i in range(n):
            ax1 = fig.add_subplot(n, 1, i+1)
            im1 = ax1.contourf(X,Y,var[:,:,i],cmap='RdYlBu_r',levels=np.linspace(-1*v_abs[i],v_abs[i],40),extend='both')
            ax1.set_ylabel('latitude / $\circ$')
            if i==0: plt.title('Field $v$ / m s$^{-1}$ on day '+str(it).zfill(3))
            if i<n-1:
                xlabels = [item.get_text() for item in ax1.get_xticklabels()]
                xempty_string_labels = [''] * len(xlabels)
                ax1.set_xticklabels(xempty_string_labels)
            else:
                ax1.set_xlabel('longitude / $\circ$')
            fig.colorbar(im1)
        plt.tight_layout()
        plt.savefig(varname+'_'+runname+'_'+str(it).zfill(4)+'.png')
        plt.clf()
if __name__ == '__main__':
    main()
