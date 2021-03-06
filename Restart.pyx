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

cdef class Restart:

    def __init__(self, Parameters Pr, namelist):
        Pr.restart        = namelist['initialize']['restart']
        Pr.restart_folder = namelist['initialize']['restart folder']
        Pr.restart_type   = namelist['initialize']['restart type']

        if Pr.restart:
            if os.path.isdir(Pr.restart_folder)==False:
                sys.exit("Restart folder does not exists")
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV,  TimeStepping TS, namelist):

        # LOAD TIME AND COMPLETE TO WHAT EVER IS IN NAMELIST

        if Pr.restart_type == '3D_output':
            # load simualtion time max from stats file and use this is the name
            data = nc.Dataset(os.getcwd() + Pr.restart_folder + 'stats/Stats.HeldSuarez.nc', 'r')
            t = data.groups['zonal_mean'].variables['t']
            if TS.t_max<=t:
                sys.exit("Restart simulation is shorter the original simulation")
            timestring = str(t)
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
            # load stats file
            ncfile = os.getcwd() + Pr.restart_folder + 'stats/Stats.HeldSuarez.nc'
            data = nc.Dataset(ncfile, 'r')
            t = data.groups['zonal_mean'].variables['t']
            if TS.t_max<=t:
                sys.exit("Restart simulation is shorter the original simulation")
            # initialize the zonal mean
            PV.P.values          = np.array(data.groups['zonal_mean'].variables['zonal_mean_P'])
            PV.T.values          = np.array(data.groups['zonal_mean'].variables['zonal_mean_T'])
            PV.QT.values         = np.array(data.groups['zonal_mean'].variables['zonal_mean_QT'])
            PV.Vorticity.values  = np.array(data.groups['zonal_mean'].variables['zonal_mean_vorticity'])
            PV.Divergence.values = np.array(data.groups['zonal_mean'].variables['zonal_mean_divergence'])
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
