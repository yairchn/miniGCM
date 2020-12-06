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
import os

class Simulation:

    def __init__(self, namelist):
        # define the member classes
        self.Gr = Grid.Grid(namelist)
        # self.TH = Thermodynamics(namelist)
        self.PV = PrognosticVariables(self.Gr)
        self.DV = DiagnosticVariables(self.Gr)
        self.Case = CasesFactory(namelist, self.Gr)
        self.DF = NumericalDiffusion()
        self.TS = TimeStepping(namelist)
        self.Stats = Stats(namelist, self.Gr)
        return

    def initialize(self, namelist):
        #initialize via Case
        self.PV.initialize(self.Gr)
        self.DV.initialize(self.Gr,self.PV)
        self.Case.initialize_forcing(self.Gr, self.PV, self.DV, namelist)
        self.Case.initialize_surface(self.Gr, namelist)
        self.DF.initialize(self.Gr, self.TS, namelist)
        self.TS.initialize(self.Gr, self.PV, self.DV, self.DF, namelist)
        self.initialize_io()
        self.io()
        # self.get_parameters(namelist)

    def run(self, namelist):
        print('run')
        while self.TS.t <= self.TS.t_max:
            self.PV.reset_pressures(self.Gr)
            self.PV.spectral_to_physical(self.Gr)
            self.DV.update(self.Gr, self.PV)
            self.DV.physical_to_spectral(self.Gr)
            # self.Case.update_surface(self.TS)
            self.Case.update_forcing(self.TS, self.Gr, self.PV, self.DV, namelist)
            self.PV.compute_tendencies(self.Gr, self.PV, self.DV, namelist)
            self.TS.update(self.Gr, self.PV, self.DV, self.DF, namelist)

            if np.mod(self.TS.t, self.Stats.stats_frequency) == 0:
                self.stats_io()
            if np.mod(self.TS.t, self.Stats.output_frequency) == 0:
                #write to stdoutput
                print('elapsed time [days] about', np.floor_divide(self.TS.t,(24.0*3600.0)))
                #write logfile
                uuid=namelist['meta']['uuid']
                outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                                   + uuid[len(uuid )-5:len(uuid)]))
                casename = namelist['meta']['casename']
                logfile=outpath+'/'+casename+'.log'
                print('logfile ',logfile)
                os.system('echo elapsed time [days] about '+str(np.floor_divide(self.TS.t,(24.0*3600.0))) + '>> '+logfile)
                os.system('echo u layer 1 min max ' + str(self.DV.U.values[:,:,0].min()) + ' ' + str(self.DV.U.values[:,:,0].max()) + '>> '+logfile)
                os.system('echo u layer 2 min max ' + str(self.DV.U.values[:,:,1].min()) + ' ' + str(self.DV.U.values[:,:,1].max()) + '>> '+logfile)
                os.system('echo u layer 3 min max ' + str(self.DV.U.values[:,:,2].min()) + ' ' + str(self.DV.U.values[:,:,2].max()) + '>> '+logfile)
                #write netcdf output
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
        self.dt   = 100.0 # timestep = 50 or 100sec (make sure it satisfies CFL)
        self.ii   = 0.0  # counter for plotting
        self.jj   = 0.0  # counter for profile loop
        self.t_next = 0.0
        return
