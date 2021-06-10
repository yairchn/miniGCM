import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/contour_surface_zonal_mean.py  QT_SurfaceFlux
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("varname")
    args = parser.parse_args()
    varname = args.varname

    folder = os.getcwd() + '/Output.HeldSuarezMoist.s0_ac10k_x10/stats/'
    ncfile = folder + 'Stats.HeldSuarezMoist.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['surface_zonal_mean'].variables[varname])
    # RainRate = np.array(data.groups['surface_zonal_mean'].variables['zonal_mean_RainRate'])
    # zonal_mean_dQTdt = np.array(data.groups['zonal_mean'].variables['zonal_mean_dQTdt'])
    # zonal_mean_QT = np.array(data.groups['zonal_mean'].variables['zonal_mean_QT'])
    t = np.divide(data.groups['surface_zonal_mean'].variables['t'],3600.0*24.0)

    X, Y = np.meshgrid(t,lat_list)
    fig = plt.figure(varname)
    plt.contourf(X,Y,np.fliplr(np.rot90(np.squeeze(var), k=3)),cmap='viridis')
    plt.xlabel('time [days]$')
    plt.ylabel('latitude / $\circ$')
    plt.colorbar()
    plt.figure('global mean')
    plt.plot(t,np.mean(var, axis =1))
    plt.xlabel('time [days]')
    plt.ylabel('surface QT flux [kg/(sec*kg)]')
    # plt.figure('dQTdt')
    # plt.plot(t,np.mean(np.mean(zonal_mean_dQTdt, axis =1), axis =1))
    # plt.xlabel('time [days]')
    # plt.ylabel('dQTdt [kg/(sec*kg)]')
    # plt.figure('QT')
    # plt.plot(t,np.mean(np.mean(zonal_mean_QT, axis =1), axis =1))
    # plt.xlabel('time [days]')
    # plt.ylabel('QT [kg/kg]')
    plt.show()
if __name__ == '__main__':
    main()
