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
    varname = 'zonal_mean_Wp'
    #runname='43a12'
    #runname='5b4838ca0516'
    folder = args.folder

    #ncfile = '/home/josefs/miniGCM/' + folder + '/stats/Stats.HeldSuarez.nc'
    ncfile = '/home/scoty/miniGCM/' + folder + '/stats/Stats.HeldSuarez.nc'
    #ncfile = '/home/yair/' + folder + '/stats/Stats.HeldSuarezMoist.nc'
    data = nc.Dataset(ncfile, 'r')


    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)
    t1 = 300
    t2 = 500

    fig,ax = plt.subplots(1,8, sharey='all',figsize=(16, 7),squeeze=False)

    i = 0
    li=[2,2,2,2,2,2,2,2]
    al=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
    la=['200 hPa','300 hPa','400 hPa','500 hPa','600 hPa','700 hPa','800 hPa','900 hPa']
    #la=['500 hPa','850 hPa', '$p_s$']
    #la=['500 hPa','750 hPa', '$p_s$']
    for j in range(8):
        ax[i,j].plot(np.mean(var[t1:t2,:,j],axis=0),np.array(lat_list),'-k',linewidth=li[j],alpha=al[j],label=la[j])
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
        ax[i,j].set_yticks(np.linspace(-90,90,7))
        ax[i,j].set_ylim(-90,90)
        ax[i,j].set_xlim(-1.5e-1,1.5e-1)
        ax[i,j].axvline(x=0,color='k',linestyle='dashed',zorder=25,alpha=0.75)
        ax[i,j].ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax[i,j].legend(loc='upper left')

    ax[0,0].set_ylabel('Latitude / $\circ$',size='12', fontname = 'Dejavu Sans')

    plt.suptitle("Vertical Velocity  / Pa s$^{-1}$",size='14', fontname = 'Dejavu Sans')
    #plt.savefig('Wp_5layers_'+runname+'.pdf', bbox_inches='tight',dpi=150)
    plt.savefig('Wp_'+folder+'.pdf', bbox_inches='tight',dpi=150)
if __name__ == '__main__':
    main()
