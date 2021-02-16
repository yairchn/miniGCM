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
    varname = 'zonal_mean_T'
    varname2= 'zonal_mean_T_eq'
    runname='410k2'
    #runname='4e5k0'
    runname='009mi'

    folder = '/home/scoty/miniGCM/Output.HeldSuarez.'+runname+'/stats/'
    ncfile = folder + 'Stats.HeldSuarez.nc'
    data = nc.Dataset(ncfile, 'r')


    runname2='007e8'
    folder2= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname2+'/stats/'
    ncfile2= folder2+ 'Stats.HeldSuarez.Restart_1.nc'
    print('ncfile2: ',ncfile2)
    data2= nc.Dataset(ncfile2,'r')

    runname3='010mi'
    folder3= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname3+'/stats/'
    ncfile3= folder3+ 'Stats.HeldSuarez.nc'
    print('ncfile3: ',ncfile3)
    data3= nc.Dataset(ncfile3,'r')

    runname4='012mi'
    folder4= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname4+'/stats/'
    ncfile4= folder4+ 'Stats.HeldSuarez.nc'
    print('ncfile4: ',ncfile4)
    data4= nc.Dataset(ncfile4,'r')

    runname5='014mi'
    folder5= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname5+'/stats/'
    ncfile5= folder5+ 'Stats.HeldSuarez.nc'
    print('ncfile5: ',ncfile5)
    data5= nc.Dataset(ncfile5,'r')

    runname6='015mi'
    folder6= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname6+'/stats/'
    ncfile6= folder6+ 'Stats.HeldSuarez.nc'
    print('ncfile6: ',ncfile6)
    data6= nc.Dataset(ncfile6,'r')

    runname7='016mi'
    folder7= '/home/scoty/miniGCM/Output.HeldSuarez.'+runname7+'/stats/'
    ncfile7= folder7+ 'Stats.HeldSuarez.nc'
    print('ncfile6: ',ncfile7)
    data7= nc.Dataset(ncfile7,'r')


    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['zonal_mean'].variables[varname])
    var2= np.array(data.groups['zonal_mean'].variables[varname2])
    t = np.divide(data.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var3= np.array(data2.groups['zonal_mean'].variables[varname])
    var4= np.array(data2.groups['zonal_mean'].variables[varname2])
    t3= np.divide(data2.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var5= np.array(data3.groups['zonal_mean'].variables[varname])
    var6= np.array(data3.groups['zonal_mean'].variables[varname2])
    t5= np.divide(data3.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var7= np.array(data4.groups['zonal_mean'].variables[varname])
    var8= np.array(data4.groups['zonal_mean'].variables[varname2])
    t7= np.divide(data4.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var9= np.array(data5.groups['zonal_mean'].variables[varname])
    var10= np.array(data5.groups['zonal_mean'].variables[varname2])
    t9= np.divide(data5.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var11= np.array(data6.groups['zonal_mean'].variables[varname])
    var12= np.array(data6.groups['zonal_mean'].variables[varname2])
    t11= np.divide(data6.groups['zonal_mean'].variables['t'],3600.0*24.0)

    var13= np.array(data7.groups['zonal_mean'].variables[varname])
    var14= np.array(data7.groups['zonal_mean'].variables[varname2])
    t13= np.divide(data7.groups['zonal_mean'].variables['t'],3600.0*24.0)


    fig = plt.figure(varname,figsize=(3,6))
    for i in range(n):
        print('i',i)
        print('min max np.mean(var[50:100,:,i],axis=0) ', np.amin(np.mean(var[50:100,:,i],axis=0)), np.amax(np.mean(var[50:100,:,i],axis=0)))
        print('lat_list.shape',lat_list.shape)
        print('var.shape',var.shape)
        ax1 = fig.add_subplot(n, 1, i+1)
        im2 = ax1.plot(np.mean(var2[50:100,:,i],axis=0),np.array(lat_list),'-r',alpha=0.4,linewidth=2)
        im1 = ax1.plot(np.mean(var[50:100,:,i],axis=0),np.array(lat_list),'-k')
        im2 = ax1.plot(np.mean(var4[200:800,:,i],axis=0),np.array(lat_list),'--',color='magenta',alpha=0.4,linewidth=2)
        im1 = ax1.plot(np.mean(var3[200:800,:,i],axis=0),np.array(lat_list),'-g')
        im2 = ax1.plot(np.mean(var6[50:100,:,i],axis=0),np.array(lat_list),'--',color='cyan',alpha=0.4,linewidth=2)
        im1 = ax1.plot(np.mean(var5[50:100,:,i],axis=0),np.array(lat_list),'-b')
        im2 = ax1.plot(np.mean(var8[90:100,:,i],axis=0),np.array(lat_list),'-',color='yellow',alpha=0.8,linewidth=0.5)
        im1 = ax1.plot(np.mean(var7[90:100,:,i],axis=0),np.array(lat_list),color='orange')
        im2 = ax1.plot(np.mean(var10[90:100,:,i],axis=0),np.array(lat_list),'-',color='gray',alpha=0.8,linewidth=0.5)
        im1 = ax1.plot(np.mean(var9[90:100,:,i],axis=0),np.array(lat_list),color='lightgreen')
        im2 = ax1.plot(np.mean(var12[90:100,:,i],axis=0),np.array(lat_list),'-',color='black',alpha=0.8,linewidth=0.5)
        im1 = ax1.plot(np.mean(var11[90:100,:,i],axis=0),np.array(lat_list),color='pink')
        im2 = ax1.plot(np.mean(var14[90:100,:,i],axis=0),np.array(lat_list),'-',color='gray',alpha=0.4,linewidth=0.2)
        im1 = ax1.plot(np.mean(var13[90:100,:,i],axis=0),np.array(lat_list),color='darkblue')
        ax1.set_ylabel('latitude / $\circ$')
        plt.grid(linestyle=':',alpha=0.6,linewidth=1)
        if i==0:
           plt.title("Temperature")
           ax1.set_xlim(200,250)
        if i==1: 
           ax1.set_xlim(220,300)
        if i==n-1:
           ax1.set_xlim(240,320)
           ax1.set_xlabel("$\overline{T}$ / K")
        plt.tight_layout()
    plt.savefig('t_3layers_'+runname+'.png')
if __name__ == '__main__':
    main()
