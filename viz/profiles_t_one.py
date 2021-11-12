import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/profiles_t_one.py varname
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("folder")
    args = parser.parse_args()
    varname = 'zonal_mean_T'
    varname2= 'zonal_mean_T_eq'
    #runname='43a12'
    #runname='5b4838ca0516'
    folder = args.folder

    ncfile = '/home/scoty/miniGCM/' + folder + '/stats/Stats.HeldSuarez.nc'
    #ncfile = '/home/yair/' + folder + '/stats/Stats.HeldSuarezMoist.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    var2= np.array(data.groups['zonal_mean'].variables[varname2])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)
    t1 = 300
    t2 = 500


    i = 0
    li=[2,2,2,2,2,2,2,2]
    al=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
    #la=['200 hPa','300 hPa','400 hPa','500 hPa','600 hPa','700 hPa','800 hPa','900 hPa']
    la=['250 hPa','500 hPa','750 hPa']
    fig,ax = plt.subplots(1,len(la), sharey='all',figsize=(16, 7),squeeze=False)
    for j in range(0,3):
        ax[i,j].plot(np.mean(var[t1:t2,:,j],axis=0),np.array(lat_list),'-k',linewidth=li[j],alpha=al[j],label=la[j])
        ax[i,j].plot(np.mean(var2[t1:t2,:,j],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
        ax[i,j].set_yticks(np.linspace(-90,90,7))
        ax[i,j].set_ylim(-90,90)
        ax[i,j].legend(loc='upper left')
    ax[0,0].set_ylabel('Latitude / $\circ$')
    #plt.xlim(-8,25)
    plt.grid(linestyle=':',alpha=0.6,linewidth=1)
    #plt.suptitle("Potential Temperature / K",size='14', fontname = 'Dejavu Sans')
    plt.suptitle("Temperature / K",size='14', fontname = 'Dejavu Sans')
    plt.tight_layout()
    #plt.savefig('Theta_'+runname+'.png',dpi=150)
    plt.savefig('T_3l_'+folder+'.pdf',dpi=150)
if __name__ == '__main__':
    main()
