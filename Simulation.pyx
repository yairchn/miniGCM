import cython
from Cases import CasesFactory
from Cases import  CasesBase
from DiagnosticVariables import DiagnosticVariables, DiagnosticVariable
from Diffusion import Diffusion
cimport Grid
import numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable
import sys
import time
import Thermodynamics
from TimeStepping import TimeStepping

class Simulation:

    def __init__(self, namelist):
        # define the member classes
        self.Gr = Grid.Grid(namelist)
        # self.TH = Thermodynamics(namelist)
        self.DV = DiagnosticVariables(self.Gr)
        self.PV = PrognosticVariables(self.Gr)
        self.Case = CasesFactory(namelist, self.Gr)
        self.DF = Diffusion()
        self.TS = TimeStepping(namelist)
        self.Stats = NetCDFIO_Stats(namelist, self.Gr)
        return

    def initialize(self, namelist):
        # self.Gr.initialize(namelist)
        # print("self.Gr.initialize(namelist)")
        self.DV.initialize(self.Gr)
        self.PV.initialize(self.Gr)
        self.Case.initialize_forcing(self.Gr, self.PV, self.DV, namelist)
        self.Case.initialize_surface(self.Gr, self.PV, self.DV, namelist)
        self.DF.initialize(self.Gr, namelist)
        self.TS.initialize(self.Gr, self.PV, self.DV, self.DF, namelist)
        self.initialize_io()
        print("self.initialize_io()")
        self.io()
        print("self.io()")

    def run(self, namelist):
        while self.TS.t <= self.TS.t_max:
            self.PV.reset_pressures(self.Gr)
            self.PV.spectral_to_physical(self.Gr)
            self.DV.update(self.Gr, self.PV)
            self.DV.physical_to_spectral(self.Gr)
            self.Case.update_surface(self.TS)
            self.Case.update_forcing(self.TS, self.Gr, self.PV, self.DV, namelist)
            # move this into timestepping-adams bashford
            self.PV.compute_tendencies(self.Gr, self.PV, self.DV, namelist)
            self.TS.update(self.Gr, self.PV, self.DV, self.DF, namelist)
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
        self.Stats.write_simulation_time(self.TS.t)
        self.DV.stats_io(self.TS, self.Stats)
        self.PV.stats_io(self.TS, self.Stats)
        self.Case.Fo.stats_io(self.TS, self.Stats)
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
