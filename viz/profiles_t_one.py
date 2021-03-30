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
    runname='ftpgne60n256'
    runname='ftpgne.1n256'
    #runname='f2800e.1n256'
    runname='theta_grid_efold100'
    #runname='theta_grid_efold10'
    #runname='theta_grid_efold1000'
    runname='-theat_p3_750hpa'
    runname='-temp__p3_750hpa'
    # folder = os.getcwd() + '/Output.HeldSuarez.'+runname+'/stats/'
    folder = '/home/yair/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    var2= np.array(data.groups['zonal_mean'].variables[varname2])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)
    t1 = 300
    t2 = 500

    #fig= plt.figure(figsize=(4,4))
    #im1 = plt.plot(np.mean(var[t1:t2,:,0],axis=0),np.array(lat_list),'-k',linewidth=4,alpha=0.4,label='top')
    #im2 = plt.plot(np.mean(var[t1:t2,:,1],axis=0),np.array(lat_list),'-k',linewidth=2,alpha=0.6,label='middle')
    #im3 = plt.plot(np.mean(var[t1:t2,:,2],axis=0),np.array(lat_list),'-k',linewidth=1,alpha=0.8,label='bottom')
    #im4 = plt.plot(np.mean(var[t1:t2,:,3],axis=0),np.array(lat_list),'-k',linewidth=.5,alpha=0.9,label='bottom')
    #im5 = plt.plot(np.mean(var[t1:t2,:,4],axis=0),np.array(lat_list),'-k',linewidth=.5,alpha=1.0,label='bottom')
    #im1a= plt.plot(np.mean(var2[t1:t2,:,0],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    #im2a= plt.plot(np.mean(var2[t1:t2,:,1],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    #im3a= plt.plot(np.mean(var2[t1:t2,:,2],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    #im4a= plt.plot(np.mean(var2[t1:t2,:,3],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    #im5a= plt.plot(np.mean(var2[t1:t2,:,4],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1)
    #plt.ylabel('Latitude / $\circ$')
    #plt.xlim(-8,25)
    #plt.grid(linestyle=':',alpha=0.6,linewidth=1)
    #plt.title("Temperature")
    #plt.xlabel("$\overline{T}$ / K")
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig('T_later_5layers_'+runname+'.png',dpi=150)
    fig,ax = plt.subplots(1,3, sharey='all',figsize=(16, 7),squeeze=False)
    i = 0
    li=[4,3,2,1,.5]
    al=[0.4,0.6,0.8,0.9,1.]
    #la=['125 hPa','250 hPa','500 hPa','750 hPa','850 hPa']
    #la=['250 hPa','500 hPa','850 hPa']
    la=['250 hPa','500 hPa','750 hPa']
    for j in range(3):
        ax[i,j].plot(np.mean(var[t1:t2,:,j],axis=0),np.array(lat_list),'-k',linewidth=li[j],alpha=al[j],label=la[j])
        ax[i,j].plot(np.mean(var2[t1:t2,:,j],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=1,label='$T_{ref}$')
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
        ax[i,j].set_yticks(np.linspace(-90,90,7))
        ax[i,j].set_ylim(-90,90)
        #ax[i,j].set_xlim(-1,1)
        #ax[i,j].axvline(x=0,color='k',linestyle='dashed',zorder=25,alpha=0.75)
        #ax[i,j].ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax[i,j].legend(loc='upper left')

    ax[0,0].set_ylabel('Latitude / $\circ$',size='12', fontname = 'Dejavu Sans')

    plt.suptitle("Potential Temperature / K",size='14', fontname = 'Dejavu Sans')
    plt.savefig('T_3layers_'+runname+'.pdf', bbox_inches='tight',dpi=150)

if __name__ == '__main__':
    main()
