from Cases import CasesFactory
from Cases import  CasesBase
import cython
from DiagnosticVariables import DiagnosticVariables, DiagnosticVariable
import Grid
import numpy as np
from NetCDFIO import NetCDFIO_Stats
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables, PrognosticVariable
import sys
import time
# import Thermodynamics
from TimeStepping import TimeStepping

class Simulation:

    def __init__(self, namelist):
        # define the member classes
        self.Gr = Grid.Grid(namelist)
        self.DV = DiagnosticVariables(self.Gr)
        self.PV = PrognosticVariables(self.Gr)
        self.Case = CasesFactory(namelist, self.Gr)
        self.TS = TimeStepping(namelist)
        self.Stats = NetCDFIO_Stats(namelist, self.Gr)
        return

    def initialize(self, namelist):
        #initialize via Case
        self.DV.initialize(self.Gr)
        self.PV.initialize(self.Gr,self.DV)
        self.Case.initialize_forcing(self.Gr, self.PV, self.DV, namelist)
        self.Case.initialize_surface(self.Gr, namelist)
        self.TS.initialize(self.Gr, self.PV, self.DV, self.DF, namelist)
        self.initialize_io()
        self.io()

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