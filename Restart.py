import numpy as np
import os
import sys
import glob
import netCDF4 as nc
from scipy.interpolate import interp2d
from Grid import Grid
from PrognosticVariables import PrognosticVariables
from Parameters import Parameters
import sphericalForcing as spf
from TimeStepping import TimeStepping

class Restart:
    def __init__(self, Pr, namelist):

        if Pr.restart:
            if os.path.isdir(os.getcwd() + Pr.restart_folder)==False:
                print('Restart folder: ', os.getcwd() + Pr.restart_folder)
                sys.exit("Restart folder does not exists")
        return

    def initialize(self, Pr, Gr, PV,  TS, namelist):
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers

        data = nc.Dataset(Pr.path_source_file, 'r')

        if Pr.restart_type == '3D_output':
            timestring = Pr.restart_time
            filepath = Pr.input_folder
            old_lat = np.array(data.groups['coordinates'].variables['latitude'])[:,1]
            old_lon = np.array(data.groups['coordinates'].variables['longitude'])[1,:]

            if os.path.exists(filepath + 'Temperature_' + timestring+'.nc'):
                TS.t = float(Pr.restart_time)
                print("Restarting simulation from time = ", TS.t)
            else:
                sys.exit("Restart files do not match exiting files: " + filepath + 'Temperature_' + timestring+'.nc')

            Pressure             = np.array(nc.Dataset(filepath + 'Pressure_' + timestring+'.nc', 'r').variables['Pressure'])
            Temperature          = np.array(nc.Dataset(filepath + 'Temperature_' + timestring+'.nc', 'r').variables['Temperature'])
            Vorticity            = np.array(nc.Dataset(filepath + 'Vorticity_' + timestring+'.nc', 'r').variables['Vorticity'])
            Divergence           = np.array(nc.Dataset(filepath + 'Divergence_' + timestring+'.nc', 'r').variables['Divergence'])
            if Pr.moist_index > 0.0:
                Specific_humidity = np.array(nc.Dataset(filepath + 'Specific_humidity_' + timestring+'.nc', 'r').variables['Specific_humidity'])

            f_Pressure = interp2d(old_lon, old_lat, Pressure, kind='linear')
            PV.P.values[:,:,nl]    = f_Pressure(Gr.longitude_list,Gr.latitude_list)
            for k in range(Pr.n_layers):
                f_Temperature = interp2d(old_lon, old_lat, Temperature[:,:,k], kind='linear')
                f_Vorticity   = interp2d(old_lon, old_lat, Vorticity[:,:,k], kind='linear')
                f_Divergence  = interp2d(old_lon, old_lat, Divergence[:,:,k], kind='linear')
                PV.T.values[:,:,k]          = f_Temperature(Gr.longitude_list,Gr.latitude_list)
                PV.Vorticity.values[:,:,k]  = f_Vorticity(Gr.longitude_list,Gr.latitude_list)
                PV.Divergence.values[:,:,k] = f_Divergence(Gr.longitude_list,Gr.latitude_list)
                if Pr.moist_index > 0.0:
                    f_Specific_humidity = interp2d(old_lon, old_lat, Specific_humidity[:,:,k], kind='linear')
                    PV.QT.values[:,:,k] = f_Specific_humidity(Gr.longitude_list,Gr.latitude_list)

        elif Pr.restart_type == 'zonal_mean':
            print(np.max(data.groups['zonal_mean'].variables['t']))
            TS.t = np.max(data.groups['zonal_mean'].variables['t'])
            if TS.t_max<=TS.t:
                print(TS.t_max,TS.t)
                sys.exit("Restart simulation is not longer the original simulation")
            for i in range(Pr.nlats):
                for j in range(Pr.nlons):
                    PV.P.values[i,j,Pr.n_layers]    = np.array(data.groups['surface_zonal_mean'].variables['zonal_mean_Ps'])[-1,i]
                    for k in range(Pr.n_layers):
                        PV.P.values[i,j,k]          = Pr.pressure_levels[k]
                        PV.T.values[i,j,k]          = np.array(data.groups['zonal_mean'].variables['zonal_mean_T'])[-1,i,k]
                        PV.Vorticity.values[i,j,k]  = np.array(data.groups['zonal_mean'].variables['zonal_mean_vorticity'])[-1,i,k]
                        PV.Divergence.values[i,j,k] = np.array(data.groups['zonal_mean'].variables['zonal_mean_divergence'])[-1,i,k]
                        if Pr.moist_index > 0.0:
                            PV.QT.values[i,j,k]     = np.array(data.groups['zonal_mean'].variables['zonal_mean_QT'])[-1,i,k]
        return