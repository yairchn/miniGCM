import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from Cases import CasesFactory
from Cases import CaseBase
from DiagnosticVariables cimport DiagnosticVariables, DiagnosticVariable
from DiagnosticVariables import DiagnosticVariables, DiagnosticVariable
from Diffusion import Diffusion
cimport Parameters
from Restart cimport Restart
cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable
from PrognosticVariables import PrognosticVariables, PrognosticVariable
import time
from TimeStepping import TimeStepping
from Microphysics import MicrophysicsBase
from LogFile import LogFile

class Simulation:

    def __init__(self, namelist):
        self.Pr = Parameters.Parameters(namelist)
        self.LF = LogFile(namelist)
        self.RS = Restart(self.Pr, namelist)
        self.Gr = Grid.Grid(self.Pr, namelist)
        self.Case = CasesFactory(namelist)
        self.PV = PrognosticVariables(self.Pr, self.Gr, namelist)
        self.DV = DiagnosticVariables(self.Pr, self.Gr)
        self.DF = Diffusion()
        self.TS = TimeStepping(namelist)
        self.Stats = NetCDFIO_Stats(self.Pr, self.Gr, namelist)
        return

    def initialize(self, namelist):
        self.LF.initialize(self.Pr, namelist)
        self.Case.initialize(self.RS, self.Pr, self.Gr, self.PV, self.TS, namelist)
        self.Case.initialize_microphysics(self.Pr, self.PV, self.DV, namelist)
        self.Case.initialize_forcing(self.Pr, self.Gr, namelist)
        self.Case.initialize_surface(self.Pr, self.Gr, self.PV, namelist)
        self.Case.initialize_convection(self.Pr, self.Gr, self.PV, namelist)
        self.DF.initialize(self.Pr, self.Gr, namelist)
        self.TS.initialize(self.Pr)
        self.initialize_io(namelist)
        self.io()

    def run(self, namelist):
        print('run')
        start_time = time.time()
        while self.TS.t <= self.TS.t_max:
            # time0 = time.time()
            self.PV.reset_pressures_and_bcs(self.Pr, self.DV)
            # print('PV.reset_pressures_and_bcs', time.time() - time0)
            # time0 = time.time()
            self.PV.spectral_to_physical(self.Pr, self.Gr)
            # print('PV.spectral_to_physical', time.time() - time0)
            # time0 = time.time()
            self.DV.update(self.Pr, self.Gr, self.PV)
            # print('DV.update', time.time() - time0)
            # time0 = time.time()
            self.Case.update(self.Pr, self.Gr, self.PV, self.DV, self.TS)
            # print('Case.update', time.time() - time0)
            # time0 = time.time()
            self.PV.compute_tendencies(self.Pr, self.Gr, self.PV, self.DV)
            # print('PV.compute_tendencies', time.time() - time0)
            # time0 = time.time()
            self.TS.update(self.Pr, self.Gr, self.PV, self.DV, self.DF, namelist)
            # print('TS.update', time.time() - time0)
            # time0 = time.time()

            if (self.TS.t%self.Stats.stats_frequency < self.TS.dt or self.TS.t == self.TS.t_max):
                wallclocktime = time.time() - start_time
                self.LF.update(self.Pr, self.TS, self.DV, self.PV, wallclocktime)
                self.stats_io()
            if (self.TS.t%self.Stats.output_frequency < self.TS.dt or self.TS.t == self.TS.t_max):
                self.io()
        return

    def initialize_io(self, namelist):
        if not namelist['initialize']['restart']:
            self.DV.initialize_io(self.Stats)
            self.PV.initialize_io(self.Stats)
            self.Case.initialize_io(self.Stats)
        return

    def io(self):
        self.DV.io(self.Pr, self.TS, self.Stats)
        self.PV.io(self.Pr, self.Gr, self.TS, self.Stats)
        self.Case.io(self.Pr, self.TS, self.Stats)
        return

    def stats_io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.DV.stats_io(self.Stats)
        self.PV.stats_io(self.Pr, self.Stats)
        self.Case.stats_io(self.PV, self.Stats)
        self.Stats.close_files()
        return

    def force_io(self):
        return