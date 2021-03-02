import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/profiles_t_one.py varname
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    args = parser.parse_args()
    runname = args.varname
    varname = 'zonal_mean_T'
    varname2= 'zonal_mean_T_eq'
    #runname='43a12'

    # folder = os.getcwd() + '/Output.HeldSuarez.'+runname+'/stats/'
    folder = os.getcwd() + '/Output.HeldSuarez.'+runname+'/'
    ncfile = folder + 'Stats.HeldSuarez.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    var2= np.array(data.groups['zonal_mean'].variables[varname2])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)
    t1 = 300
    t2 = 340

    fig= plt.figure(figsize=(4,4))
    im1 = plt.plot(np.mean(var[t1:t2,:,0],axis=0),np.array(lat_list),'-k',linewidth=4,alpha=0.4,label='top')
    im2 = plt.plot(np.mean(var[t1:t2,:,1],axis=0),np.array(lat_list),'-k',linewidth=2,alpha=0.6,label='middle')
    im3 = plt.plot(np.mean(var[t1:t2,:,2],axis=0),np.array(lat_list),'-k',linewidth=1,alpha=0.8,label='bottom')
    im1a= plt.plot(np.mean(var2[t1:t2,:,0],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    im2a= plt.plot(np.mean(var2[t1:t2,:,1],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    im3a= plt.plot(np.mean(var2[t1:t2,:,2],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    plt.ylabel('Latitude / $\circ$')
    #plt.xlim(-8,25)
    plt.grid(linestyle=':',alpha=0.6,linewidth=1)
    plt.title("Temperature")
    plt.xlabel("$\overline{T}$ / K")
    plt.legend()
    plt.tight_layout()
    plt.savefig('T_3layers_'+runname+'.png',dpi=150)
if __name__ == '__main__':
    main()
