
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *

from Cases import CasesFactory
from Cases import  CasesBase
import time
import Thermodynamics
from Diffusion import NumericalDiffusion
from Grid import Grid
from ReferenceState import ReferenceState
from NetCDFIO import Stats
from TimeStepping import TimeStepping
from DiagnosticVariables import DiagnosticVariables, DiagnosticVariable
from PrognosticVariables import PrognosticVariables, PrognosticVariable

class Simulation:

    def __init__(self, namelist):
        # define the member classes
        self.Gr = Grid(namelist)
        # self.TH = Thermodynamics(namelist)
        self.DV = DiagnosticVariables(self.Gr)
        self.PV = PrognosticVariables(self.Gr)
        self.Case = CasesFactory(namelist, self.Gr)
        self.DF = NumericalDiffusion()
        self.TS = TimeStepping(namelist)
        # self.RS = ReferenceState()
        # self.AB = AdamsBashforth()
        self.Stats = Stats(namelist, self.Gr)
        return

    def initialize(self, namelist):
        #initialize via Case
        self.DV.initialize(self.Gr)
        self.PV.initialize(self.Gr,self.DV)
        self.Case.initialize_forcing(self.Gr, namelist)
        self.Case.initialize_surface(self.Gr, namelist)
        self.DF.initialize(self.Gr, self.TS, namelist)
        self.TS.initialize(self.Gr, self.PV, self.DV, self.DF, namelist)
        self.initialize_io()
        self.io()

        # self.get_parameters(namelist)
        # self.RS.initialize(self.Gr)
        # self.Case.initialize_reference(self.Gr)
    def run(self, namelist):
        while self.TS.t <= self.TS.t_max:
            self.DV.physical_to_spectral(self.Gr)
            self.PV.physical_to_spectral(self.Gr)
            self.DV.update(self.Gr, self.PV)
            self.Case.update_surface(self.TS)
            self.Case.update_forcing(self.TS, self.Gr, self.PV, self.DV, namelist)
            # move this into timestepping-adams bashford 
            self.PV.compute_tendencies(self.Gr, self.PV, self.DV, namelist)
            self.TS.update(self.Gr, self.PV, self.DV, self.DF, namelist)
            self.PV.spectral_to_physical(self.Gr)
            print('----------------------------------------- time =', self.TS.t)
            # plt.figure('surface pressure field')
            # plt.contourf(self.PV.P.values[:,:,3])
            # plt.colorbar()
            # plt.show()
            for k in range(3):
                print(k,'PV.P.values',np.min(self.PV.P.values[:,:,k]),np.max(self.PV.P.values[:,:,k]))
                print(k,'PV.Divergence.values',np.min(self.PV.Divergence.values[k]),np.max(self.PV.Divergence.values[k]))
                print(k,'PV.Vorticity.values',np.min(self.PV.Vorticity.values[:,:,k]),np.max(self.PV.Vorticity.values[:,:,k]))
                print(k,'PV.Divergence.tendency',np.min(self.PV.Divergence.tendency[:,k]),np.max(self.PV.Divergence.tendency[:,k]))
                print(k,'PV.Vorticity.tendency',np.min(self.PV.Vorticity.tendency[:,k]),np.max(self.PV.Vorticity.tendency[:,k]))
                print(k,'PV.T.values',np.min(self.PV.T.values[:,:,k]),np.max(self.PV.T.values[:,:,k]))
                # plt.show()

            print(3,'PV.P.values',np.min(self.PV.P.values[1,:,3]),np.max(self.PV.P.values[1,:,3]))
            # Apply the tendencies, also update the BCs and diagnostic thermodynamics
            if np.mod(self.TS.t, self.Stats.stats_frequency) == 0:
                self.stats_io()
            if np.mod(self.TS.t, self.Stats.output_frequency) == 0:
                self.io()
        return

    def initialize_io(self):

        self.DV.initialize_io(self.Stats)
        self.PV.initialize_io(self.Stats)
        self.Case.Fo.initialize_io(self.Stats)
        self.Case.Sur.initialize_io(self.Stats)
        return

    def io(self):
        self.DV.io(self.Gr, self.TS, self.Stats)
        self.PV.io(self.Gr, self.TS, self.Stats)
        self.Case.Fo.io(self.Gr, self.TS, self.Stats)
        self.Case.io(self.PV, self.Gr, self.TS, self.Stats)
        return

    def stats_io(self):
        self.Stats.open_files()
        # YAIR - move all write_3D_data to here and concentrate all vars in a single file per times step 
        self.DV.stats_io(self.TS, self.Stats)
        self.PV.stats_io(self.TS, self.Stats)
        self.Stats.close_files()
        return

    def force_io(self):
        return

    def get_parameters(self, namelist):
        # move all parameters to namelist default
        # parameters = OrderedDict()

        ##################################################################
        # may be run for even longer time

        # what are all of these ?
        self.t    = 0.0 # initial time = 0
        self.t3   = 0.0 # initial time = 0
        self.t2   = 0.0 # initial time = 0
        self.t1   = 0.0 # initial time = 0
        self.dt   = 50.0 # timestep = 50 or 100sec (make sure it satisfies CFL)
        self.dt   =100.0 # timestep = 50 or 100sec (make sure it satisfies CFL)
        self.ii   = 0.0  # counter for plotting
        self.jj   = 0.0  # counter for profile loop
        self.t_next = 0.0
        # setting up the integration
        self.tmax =   10.01*24.0*3600.0  #(time to integrate, here 1000 days)
        self.tmax =   90.01*24.0*3600.0  #(time to integrate, here 1000 days)
        self.tmax = 1000.01*24.0*3600.0  #(time to integrate, here 1000 days)
        ################################################################
        # Logging the data
        #time_step = 5*24*3600.
        #time_step =   24*3600.
        self.time_step  =    2*3600.0
        self.ndiss = namelist['diffusion']['order']
        return
