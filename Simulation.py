import numpy as np
import matplotlib.pyplot as plt
from Cases import CasesFactory
from Cases import  CasesBase
import time
import Thermodynamics
from Diffusion import NumericalDiffusion
import Grid
from ReferenceState import ReferenceState
from NetCDFIO import Stats
from TimeStepping import TimeStepping
from DiagnosticVariables import DiagnosticVariables, DiagnosticVariable
from PrognosticVariables import PrognosticVariables, PrognosticVariable
from Microphysics import MicrophysicsBase
import Parameters

class Simulation:

    def __init__(self, namelist):
        # define the member classes
        self.Pr = Parameters.Parameters(namelist)
        self.Gr = Grid.Grid(self.Pr, namelist)
        self.Case = CasesFactory(self.Pr, namelist)
        self.PV = PrognosticVariables(self.Pr, self.Gr)
        self.DV = DiagnosticVariables(self.Pr, self.Gr)
        self.DF = NumericalDiffusion()
        self.TS = TimeStepping(self.Pr, namelist)
        self.Stats = Stats(self.Pr, self.Gr, namelist)
        return

    def initialize(self, namelist):
        self.Case.initialize(self.Pr, self.Gr, self.PV, namelist)
        self.DV.initialize(self.Pr, self.Gr,self.PV)
        self.PV.initialize(self.Pr)
        self.Case.initialize_microphysics(self.Pr, namelist)
        self.Case.initialize_forcing(self.Pr)
        self.Case.initialize_surface(self.Pr, self.Gr, self.PV)
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
            self.Case.update_surface(self.Pr, self.Gr, self.TS, self.PV)
            self.Case.update_forcing(self.Pr, self.Gr, self.TS, self.PV, self.DV)
            self.Case.update_microphysics(self.Pr, self.Gr, self.PV, self.TS)
            self.PV.compute_tendencies(self.Pr, self.Gr, self.PV, self.DV, self.Case.MP)
            self.TS.update(self.Pr, self.Gr, self.PV, self.DV, self.DF, namelist)
            print('elapsed time [days] about', np.floor_divide(self.TS.t,(24.0*3600.0)))
            if np.mod(self.TS.t, self.Stats.stats_frequency) == 0:
                self.stats_io()
            if np.mod(self.TS.t, self.Stats.output_frequency) == 0:
                print('elapsed time [days] about', np.floor_divide(self.TS.t,(24.0*3600.0)))
                self.io()
        return

    def initialize_io(self):

        self.DV.initialize_io(self.Stats)
        self.PV.initialize_io(self.Stats)
        self.Case.MP.initialize_io(self.Stats)
        self.Case.Fo.initialize_io(self.Stats)
        self.Case.Sur.initialize_io(self.Stats)
        return

    def io(self):
        self.DV.io(self.Pr, self.TS, self.Stats)
        self.PV.io(self.Pr, self.TS, self.Stats)
        self.Case.MP.io(self.Pr, self.TS, self.Stats)
        self.Case.Fo.io(self.Pr, self.TS, self.Stats)
        self.Case.io(self.Pr, self.TS, self.PV, self.Stats)
        return

    def stats_io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.DV.stats_io(self.TS, self.Stats)
        self.PV.stats_io(self.TS, self.Stats)
        self.Case.MP.stats_io(self.TS, self.Stats)
        self.Case.Fo.stats_io(self.TS, self.Stats)
        self.Stats.close_files()
        return

    def force_io(self):
        return