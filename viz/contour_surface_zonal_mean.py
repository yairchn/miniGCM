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

    ncfile = os.getcwd() + '/Output.HeldSuarezMoist.evap-6layers/stats/Stats.HeldSuarezMoist.nc'
    data = nc.Dataset(ncfile, 'r')

    lat = np.array(data.groups['coordinates'].variables['latitude'])
    n = int(np.multiply(data.groups['coordinates'].variables['layers'],1.0))

    lat_list = np.array(data.groups['coordinates'].variables['latitude_list'])
    var = np.array(data.groups['surface_zonal_mean'].variables[varname])
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
    plt.show()
if __name__ == '__main__':
    main()
