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
    varname = 'zonal_mean_QT'
    #runname='43a12'
    #runname='5b4838ca0516'
    folder = args.folder

    # folder = os.getcwd() + '/Output.HeldSuarez.'+runname+'/stats/'
    #folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    #folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    #ncfile = folder + 'Stats.HeldSuarez.nc'
    ncfile = '/home/yair/' + folder + '/stats/Stats.HeldSuarezMoist.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)
    t1 = 300
    t2 = 500
    #t1 = 30
    #t2 = 80


    i = 0
    li=[4,3,2,1,.5]
    al=[0.4,0.6,0.8,0.9,1.]
    #la=['125 hPa','250 hPa','500 hPa','750 hPa','850 hPa']
    la=['250 hPa','500 hPa','750 hPa']
    #la=['250 hPa','350 hPa','450 hPa','550 hPa','650 hPa','750 hPa','850 hPa','950 hPa']
    fig,ax = plt.subplots(1,len(la), sharey='all',figsize=(16, 7),squeeze=False)
    for j in range(0,3):
        ax[i,j].plot(np.mean(var[t1:t2,:,j],axis=0)*1.e3,np.array(lat_list),'-k',linewidth=li[j],alpha=al[j],label=la[j])
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
        ax[i,j].set_yticks(np.linspace(-90,90,7))
        ax[i,j].set_ylim(-90,90)
        ax[i,j].legend(loc='upper left')
        ax[i,j].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0,0].set_ylabel('Latitude / $\circ$')
    #plt.xlim(-8,25)
    plt.grid(linestyle=':',alpha=0.6,linewidth=1)
    plt.suptitle("Zonalmean QT / g kg$^{-1}$",size='14', fontname = 'Dejavu Sans')
    plt.title('\n')
    plt.tight_layout()
    #plt.savefig('U_3l_'+runname+'.pdf',dpi=150)
    plt.savefig('QT_3l_'+folder+'.pdf',dpi=150)
if __name__ == '__main__':
    main()
