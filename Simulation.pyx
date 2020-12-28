import cython
import numpy as np
from Cases import CasesFactory
from Cases import CasesBase
from DiagnosticVariables cimport DiagnosticVariables, DiagnosticVariable
from DiagnosticVariables import DiagnosticVariables, DiagnosticVariable
from Diffusion import Diffusion
cimport Parameters
cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable
from PrognosticVariables import PrognosticVariables, PrognosticVariable
import sys
import time
from TimeStepping import TimeStepping
from TimeStepping import TimeStepping
from Microphysics import MicrophysicsBase

class Simulation:

    def __init__(self, namelist):
        # define the member classes
        self.Pr = Parameters.Parameters(namelist)
        print(dir(self.Pr))
        self.Gr = Grid.Grid(self.Pr, namelist)
        self.Case = CasesFactory(self.Pr, namelist)
        self.PV = PrognosticVariables(self.Pr, self.Gr)
        self.DV = DiagnosticVariables(self.Pr, self.Gr)
        self.Case = CasesFactory(namelist, self.Gr)
        self.DF = Diffusion()
        self.TS = TimeStepping(self.Pr, namelist)
        self.Stats = NetCDFIO_Stats(self.Pr, self.Gr, namelist)
        return

    def initialize(self, namelist):
        self.Case.initialize(self.Pr, self.Gr, self.PV, namelist)
        self.DV.initialize(self.Pr, self.Gr,self.PV)
        self.PV.initialize(self.Pr)
        self.Case.initialize_microphysics(self.Pr, self.PV, namelist)
        self.Case.initialize_forcing(self.Pr)
        self.Case.initialize_surface(self.Pr, self.Gr, self.PV, namelist)
        self.DF.initialize(self.Pr, self.Gr, self.TS, namelist)
        self.TS.initialize()
        self.initialize_io()
        self.io()

    def run(self, namelist):
        print('run')
        while self.TS.t <= self.TS.t_max:
            self.PV.reset_pressures(self.Pr, self.Gr)
            self.PV.spectral_to_physical(self.Pr, self.Gr)
            self.DV.update(self.Pr, self.Gr, self.PV)
            self.DV.physical_to_spectral(self.Pr, self.Gr)
            self.Case.update_microphysics(self.Pr, self.Gr, self.PV, self.TS)
            self.Case.update_forcing(self.Pr, self.Gr, self.TS, self.PV, self.DV)
            self.Case.update_surface(self.Pr, self.Gr, self.TS, self.PV, self.DV)
            self.PV.compute_tendencies(self.Pr, self.Gr, self.PV, self.DV, self.Case.MP)
            self.TS.update(self.Pr, self.Gr, self.PV, self.DV, self.DF, namelist)

            if np.mod(self.TS.t, self.Stats.stats_frequency) == 0:
                self.stats_io()
            if np.mod(self.TS.t, self.Stats.output_frequency) == 0:
                print('elapsed time [days] about', np.floor_divide(self.TS.t,(24.0*3600.0)))
                print('minimum surface pressure is', np.min(self.PV.P.values[:,:,self.Pr.n_layers]))
                self.io()
        return

    def initialize_io(self):

        self.DV.initialize_io(self.Stats)
        self.PV.initialize_io(self.Stats)
        self.Case.initialize_io(self.Stats)
        return

    def io(self):
        self.DV.io(self.Gr, self.TS, self.Stats)
        self.PV.io(self.Gr, self.TS, self.Stats)
        self.Case.io(self.Gr, self.TS, self.Stats)
        return

    def io(self):
        self.DV.io(self.Pr, self.TS, self.Stats)
        self.PV.io(self.Pr, self.TS, self.Stats)
        self.Case.io(self.Pr, self.TS, self.PV, self.Stats)
        return

    def stats_io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.DV.stats_io(self.TS, self.Stats)
        self.PV.stats_io(self.TS, self.Stats)
        self.Case.stats_io(self.TS, self.Stats)
        self.Stats.close_files()
        return

    def force_io(self):
        return