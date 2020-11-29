import numpy as np
from NetCDFIO import Stats
from PrognosticVariables import PrognosticVariables
from DiagnosticVariables import DiagnosticVariables
from Grid import Grid
from NetCDFIO import Stats
import TimeStepping
import Forcing
import Surface
import ReferenceState


def CasesFactory(namelist, Gr):
    if namelist['meta']['casename'] == 'HeldSuarez':
        return HeldSuarez(namelist, Gr)
    # anthoer example
    # elif namelist['meta']['casename'] == 'Stochastic_Forcing':
    #     return Stochastic_Frorcing(paramlist)
    else:
        print('case not recognized')
    return


class CasesBase:
    def __init__(self, namelist):
        return
    # def initialize_reference(self, Gr):
    #     return
    # def initialize_profiles(self):
    #     return
    def initialize_surface(self, Gr):
        return
    def initialize_forcing(self, Gr, PV, DV, namelist):
        return
    def initialize_io(self, Stats):
        return
    def io(self, PV, Gr, TS, Stats):
        return
    def update_surface(self):
        return
    def update_forcing(self):
        return

class HeldSuarez(CasesBase):
    def __init__(self, namelist, Gr):
        self.casename = 'HelzSuarez'
        self.tau = 50.0*24.0*3600.0
        self.dTdy        = 60.0
        self.T_lapserate = 10.0
        self.p_ref       = 10e5
        self.foring_type = 'relaxation'
        #Temperature profiles from Held and Suarez (1994) -- initialize isothermally in each layer
        self.Tinit  = [229.0, 257.0, 295.0] # make this a matrix size self.T.values 
        self.Fo  = Forcing.Forcing_HelzSuarez()
        self.Sur = Surface.SurfaceNone()
        self.RS = ReferenceState.ReferenceState()
        self.DV = DiagnosticVariables(Gr)
        self.PV = PrognosticVariables(Gr)

        return

    # def initialize_reference(self, Gr):
    #     self.RS.initialize(Gr)
    #     return

    def initialize_surface(self, Gr, namelist):
        self.Sur.initialize()
        return

    def initialize_forcing(self, Gr, PV, DV, namelist):
        self.Fo.initialize(Gr, PV, DV, namelist)
        return

    def initialize_io(self, Stats):
        Stats.initialize_io(self, Stats)
        return

    def io(self, PV, Gr, TS, Stats):
        CasesBase.io(self, PV, Gr, TS, Stats)
        return

    def update_reference(self, Gr):
        # self.RS.update(Gr, Stats)
        return

    def update_surface(self, Gr):
        # self.Sur.update(self)
        return

    def update_forcing(self, TS, Gr, PV, DV, namelist):
        self.Fo.update(TS, Gr, PV, DV, namelist)
        return