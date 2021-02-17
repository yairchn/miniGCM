import cython
import numpy as np
from Cases import CasesFactory
from Cases import CaseBase
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
from Microphysics import MicrophysicsBase
from LogFile import LogFile

class Simulation:

    def __init__(self, namelist):
        self.Pr = Parameters.Parameters(namelist)
        self.LF = LogFile(namelist)
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
        print('initialize Held & Suarez')
        self.Case.initialize(self.Pr, self.Gr, self.PV, self.DV, namelist)
        self.DV.initialize(self.Pr, self.Gr,self.PV)
        #self.PV.initialize(self.Pr)
        self.Case.initialize_microphysics(self.Pr, self.PV, self.DV, namelist)
        self.Case.initialize_forcing(self.Pr, namelist)
        self.Case.initialize_surface(self.Pr, self.Gr, self.PV, namelist)
        self.DF.initialize(self.Pr, self.Gr, namelist)
        self.TS.initialize()
        self.initialize_io()
        self.io()

    def run(self, namelist):
        print('run')
        start_time = time.time()
        while self.TS.t <= self.TS.t_max:
            # t0 = time.clock()
            self.PV.reset_pressures(self.Pr)
            # print('PV.reset_pressures',time.clock() - t0) # 0.002265999999998769)
            # t0 = time.clock()
            self.PV.spectral_to_physical(self.Pr, self.Gr)
            # print('PV.spectral_to_physical',time.clock() - t0) # 0.05583999999999989)
            # t0 = time.clock()
            self.DV.update(self.Pr, self.Gr, self.PV)
            # print('DV.update',time.clock() - t0) # 0.1323400000000028)
            # t0 = time.clock()
            self.DV.physical_to_spectral(self.Pr, self.Gr)
            # print('DV.physical_to_spectral',time.clock() - t0) # 0.0481999999999978)
            # t0 = time.clock()
            self.Case.update(self.Pr, self.Gr, self.PV, self.DV, self.TS)
            # print('Case.update_microphysics',time.clock() - t0) # 0.11864800000000031)
            # t0 = time.clock()
            self.PV.compute_tendencies(self.Pr, self.Gr, self.PV, self.DV)
            # print('PV.compute_tendencies',time.clock() - t0) # 0.14849600000000152)
            # t0 = time.clock()
            self.TS.update(self.Pr, self.Gr, self.PV, self.DV, self.DF, namelist)
            # print('TS.update',time.clock() - t0) # 0.07199999999999918)
            # t0 = time.clock()

            if np.mod(self.TS.t, self.Stats.stats_frequency) == 0:
                self.stats_io()
                wallclocktime = time.time() - start_time
                self.LF.update(self.Pr, self.TS, self.DV, self.PV, wallclocktime, namelist)
            if np.mod(self.TS.t, self.Stats.output_frequency) == 0:
                self.io()
        return

    def initialize_io(self):

        self.DV.initialize_io(self.Stats)
        self.PV.initialize_io(self.Stats)
        self.Case.initialize_io(self.Stats)
        return

    def io(self):
        self.DV.io(self.Pr, self.TS, self.Stats)
        self.PV.io(self.Pr, self.TS, self.Stats)
        self.Case.io(self.Pr, self.TS, self.Stats)
        return

    def stats_io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.DV.stats_io(self.Stats)
        self.PV.stats_io(self.Stats)
        self.Case.stats_io(self.PV, self.Stats)
        self.Stats.close_files()
        return

    def force_io(self):
        return
