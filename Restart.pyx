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

cdef class Restart:

    def __init__(self, Parameters Pr, namelist):

        if Pr.restart:
            if os.path.isdir(os.getcwd() + Pr.restart_folder)==False:
                print('Restart folder: ', os.getcwd() + Pr.restart_folder)
                sys.exit("Restart folder does not exists")
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV,  TimeStepping TS, namelist):
        cdef:
            double [:,:] noise
        # LOAD TIME AND COMPLETE TO WHAT EVER IS IN NAMELIST

        # load stats file
        data = nc.Dataset(Pr.path_plus_file, 'r')
        TS.t = np.max(data.groups['zonal_mean'].variables['t'])
        if TS.t_max<=TS.t:
            print(TS.t_max,TS.t)
            sys.exit("Restart simulation is not larger the original simulation")

        if Pr.restart_type == '3D_output':
            filepath = os.getcwd() + Pr.restart_folder + 'Fields/'

            P                    = nc.Dataset(filepath + 'Pressure_' + str(TS.t)+'.nc', 'r')
            PV.P.values          = np.array(data.variables['P'])
            Temperature          = nc.Dataset( filepath + 'Temperature_' + str(TS.t)+'.nc', 'r')
            PV.T.values          = np.array(data.variables['Temperature'])
            Vorticity            = nc.Dataset(filepath + 'Vorticity_' + str(TS.t)+'.nc', 'r')
            PV.Vorticity.values  = np.array(data.variables['Vorticity'])
            QT                   = nc.Dataset(filepath + 'Specific_humidity_' + str(TS.t)+'.nc', 'r')
            PV.QT.values         = np.array(data.variables['QT'])
            Divergence           = nc.Dataset(filepath + 'Divergence_' + str(TS.t)+'.nc', 'r')
            PV.Divergence.values = np.array(data.variables['Divergence'])

        elif Pr.restart_type == 'zonal_mean':
            # initialize the zonal mean
            for i in range(Pr.nlats):
                for j in range(Pr.nlons):
                    for k in range(Pr.n_layers):
                        PV.T.values[i,j,k]          = np.array(data.groups['zonal_mean'].variables['zonal_mean_T'])[-1,i,k]
                        PV.QT.values[i,j,k]         = np.array(data.groups['zonal_mean'].variables['zonal_mean_QT'])[-1,i,k]
                        PV.Vorticity.values[i,j,k]  = np.array(data.groups['zonal_mean'].variables['zonal_mean_vorticity'])[-1,i,k]
                        PV.Divergence.values[i,j,k] = np.array(data.groups['zonal_mean'].variables['zonal_mean_divergence'])[-1,i,k]
                        PV.P.values[i,j,k]          = PV.P_init[k]
                    PV.P.values[i,j,Pr.n_layers]    = np.array(data.groups['surface_zonal_mean'].variables['zonal_mean_Ps'])[-1,i]
        return
