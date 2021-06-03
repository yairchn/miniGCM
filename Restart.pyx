import cython
from Grid cimport Grid
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from Parameters cimport Parameters
import os
import sys
import glob
import netCDF4 as nc
import sphericalForcing as spf
from TimeStepping cimport TimeStepping
import pylab as plt
from scipy.interpolate import interp2d

cdef class Restart:

    def __init__(self, Parameters Pr, namelist):

        if Pr.restart:
            if os.path.isdir(os.getcwd() + Pr.restart_folder)==False:
                print('Restart folder: ', os.getcwd() + Pr.restart_folder)
                sys.exit("Restart folder does not exists")
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV,  TimeStepping TS, namelist):

        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers

        data = nc.Dataset(Pr.path_plus_file, 'r')

        if Pr.restart_type == '3D_output':
            timestring = Pr.restart_time
            filepath = Pr.input_folder
            if os.path.exists(filepath + 'Temperature_' + timestring+'.nc'):
                TS.t = float(Pr.restart_time)
                print("Restarting simulation from time = ", TS.t)
            else:
                sys.exit("Restart files do not match exiting files: " + filepath + 'Temperature_' + timestring+'.nc')

            Temperature          = nc.Dataset( filepath + 'Temperature_' + timestring+'.nc', 'r')
            PV.T.values          = np.array(Temperature.variables['Temperature'])
            Vorticity            = nc.Dataset(filepath + 'Vorticity_' + timestring+'.nc', 'r')
            PV.Vorticity.values  = np.array(Vorticity.variables['Vorticity'])
            Divergence           = nc.Dataset(filepath + 'Divergence_' + timestring+'.nc', 'r')
            PV.Divergence.values = np.array(Divergence.variables['Divergence'])
            Pressure             = nc.Dataset(filepath + 'Pressure_' + timestring+'.nc', 'r')
            PV.P.values.base[:,:,nl]  = np.array(Pressure.variables['Pressure'])
            if Pr.moist_index > 0.0:
                Specific_humidity    = nc.Dataset(filepath + 'Specific_humidity_' + timestring+'.nc', 'r')
                PV.QT.values         = np.array(Specific_humidity.variables['Specific_humidity'])

        elif Pr.restart_type == '3D_interpolate':
            timestring = Pr.restart_time
            filepath = Pr.input_folder
            old_lat = np.array(data.groups['coordinates'].variables['latitude'])[:,1]
            old_lon = np.array(data.groups['coordinates'].variables['longitude'])[1,:]

            print("file ", filepath + 'Temperature_' + timestring+'.nc')
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
            PV.P.values.base[:,:,nl]    = f_Pressure(Gr.longitude_list,Gr.latitude_list)
            for k in range(Pr.n_layers):
                f_Temperature = interp2d(old_lon, old_lat, Temperature[:,:,k], kind='linear')
                f_Vorticity   = interp2d(old_lon, old_lat, Vorticity[:,:,k], kind='linear')
                f_Divergence  = interp2d(old_lon, old_lat, Divergence[:,:,k], kind='linear')
                PV.T.values.base[:,:,k]          = f_Temperature(Gr.longitude_list,Gr.latitude_list)
                PV.Vorticity.values.base[:,:,k]  = f_Vorticity(Gr.longitude_list,Gr.latitude_list)
                PV.Divergence.values.base[:,:,k] = f_Divergence(Gr.longitude_list,Gr.latitude_list)
                if Pr.moist_index > 0.0:
                    f_Specific_humidity = interp2d(old_lon, old_lat, Specific_humidity[:,:,k], kind='linear')
                    PV.QT.values.base[:,:,k] = f_Specific_humidity(Gr.longitude_list,Gr.latitude_list)
            print("Temperature min shape ", np.min(PV.T.values), np.shape(PV.T.values))
            print("Pressure min shape ", np.min(PV.P.values), np.shape(PV.P.values))

        elif Pr.restart_type == 'zonal_mean':
            TS.t = np.max(data.groups['zonal_mean'].variables['t'])
            if TS.t_max<=TS.t:
                print(TS.t_max,TS.t)
                sys.exit("Restart simulation is not longer the original simulation")
            for i in range(Pr.nlats):
                for j in range(Pr.nlons):
                    PV.P.values[i,j,Pr.n_layers]    = np.array(data.groups['surface_zonal_mean'].variables['zonal_mean_Ps'])[-1,i]
                    for k in range(Pr.n_layers):
                        PV.P.values[i,j,k]          = PV.P_init[k]
                        PV.T.values[i,j,k]          = np.array(data.groups['zonal_mean'].variables['zonal_mean_T'])[-1,i,k]
                        PV.Vorticity.values[i,j,k]  = np.array(data.groups['zonal_mean'].variables['zonal_mean_vorticity'])[-1,i,k]
                        PV.Divergence.values[i,j,k] = np.array(data.groups['zonal_mean'].variables['zonal_mean_divergence'])[-1,i,k]
                        if Pr.moist_index > 0.0:
                            PV.QT.values[i,j,k]     = np.array(data.groups['zonal_mean'].variables['zonal_mean_QT'])[-1,i,k]
        return
