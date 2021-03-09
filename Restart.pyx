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
        # stats_file = os.getcwd() + Pr.restart_folder + '/stats/Stats.HeldSuarez.nc'
        stats_file = Pr.path_plus_file
        data = nc.Dataset(stats_file, 'r')
        TS.t = np.max(data.groups['zonal_mean'].variables['t'])
        if TS.t_max<=TS.t:
            # restart_stats_file = os.getcwd() + Pr.restart_folder + '/stats/'+Pr.path_plus_file
            # os.remove(Pr.path_plus_file)
            # shutil.copy(stats_file, restart_stats_file)
            print(TS.t_max,TS.t)
            sys.exit("Restart simulation is not larger the original simulation")

        if Pr.restart_type == '3D_output':
            # load simualtion time max from stats file and use this is the name
            timestring = str(TS.t)
            # find the last 3D file for each needed variable
            # list_of_files = [os.path.basename(x) for x in glob.glob('Output.HeldSuarez._cOG_/Fields/*')]
            # div_files = []
            # for file in list_of_files:
            #     if "Divergence" in file:
            #         div_files += [file[11:-3]]
            # timestring = str(np.max(list(map(int, div_files))))

            # initialize 3D fields and load
            ncfile = os.getcwd() + Pr.restart_folder + 'Fields/' + 'Pressure_' + timestring+'.nc'
            data = nc.Dataset(ncfile, 'r')
            PV.P.values = np.array(data.variables['P'])
            ncfile = os.getcwd() + Pr.restart_folder + 'Fields/' + 'Temperature_' + timestring+'.nc'
            data = nc.Dataset(ncfile, 'r')
            PV.T.values = np.array(data.variables['Temperature'])
            ncfile = os.getcwd() + Pr.restart_folder + 'Fields/' + 'Vorticity_' + timestring+'.nc'
            data = nc.Dataset(ncfile, 'r')
            PV.Vorticity.values = np.array(data.variables['Vorticity'])
            ncfile = os.getcwd() + Pr.restart_folder + 'Fields/' + 'Specific_humidity_' + timestring+'.nc'
            data = nc.Dataset(ncfile, 'r')
            PV.QT.values = np.array(data.variables['QT'])
            ncfile = os.getcwd() + Pr.restart_folder + 'Fields/' + 'Divergence_' + timestring+'.nc'
            data = nc.Dataset(ncfile, 'r')
            PV.Divergence.values = np.array(data.variables['Divergence'])

            PV.physical_to_spectral(Pr, Gr)

        elif Pr.restart_type == 'zonal_mean':
            # initialize the zonal mean
            P_init = np.array([Pr.p1, Pr.p2, Pr.p3])
            for i in range(Pr.nlats):
                for j in range(Pr.nlons):
                    for k in range(Pr.n_layers):
                        PV.T.values[i,j,k]          = np.array(data.groups['zonal_mean'].variables['zonal_mean_T'])[-1,i,k]
                        PV.QT.values[i,j,k]         = np.array(data.groups['zonal_mean'].variables['zonal_mean_QT'])[-1,i,k]
                        PV.Vorticity.values[i,j,k]  = np.array(data.groups['zonal_mean'].variables['zonal_mean_vorticity'])[-1,i,k]
                        PV.Divergence.values[i,j,k] = np.array(data.groups['zonal_mean'].variables['zonal_mean_divergence'])[-1,i,k]
                        PV.P.values[i,j,k]          = P_init[k]
                    PV.P.values[i,j,Pr.n_layers]    = np.array(data.groups['surface_zonal_mean'].variables['zonal_mean_Ps'])[-1,i]
                    # print(np.shape(PV.P.values[i,j,0:Pr.n_layers]))
                    # print(np.shape(np.zeros(Pr.n_layers)))
                    # # PV.P.values[i,j,0:Pr.n_layers]= np.add(np.zeros(Pr.n_layers),([Pr.p1, Pr.p2, Pr.p3]))
                    # PV.P.values[i,j,0:Pr.n_layers]= np.zeros(Pr.n_layers)
            PV.physical_to_spectral(Pr, Gr)
            # add noise to Temperature
            if Pr.inoise==1:
                 # calculate noise
                 F0=np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
                 fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0., noise_type=Pr.noise_type)
                 noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(F0))*Pr.noise_amp
                 # save noise here
                 # np.save('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy',noise)
                 # load noise here
                 # noise = np.load('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy')
                 # add noise
                 PV.T.spectral.base[:,Pr.n_layers-1] = np.add(PV.T.spectral.base[:,Pr.n_layers-1],
                                                            Gr.SphericalGrid.grdtospec(noise.base))


            return



        return
