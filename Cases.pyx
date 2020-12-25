import numpy as np
from PrognosticVariables cimport PrognosticVariables
from PrognosticVariables cimport PrognosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
cimport Forcing
cimport Surface
import cython
import sys
from TimeStepping cimport TimeStepping

def CasesFactory(namelist, Gr):
    if namelist['meta']['casename'] == 'HeldSuarez':
        return HeldSuarez(namelist)
    # elif namelist['meta']['casename'] == 'HeldSuarez_moist':
    #     return HeldSuarez_moist(namelist, Gr)
    # anthoer example
    # elif namelist['meta']['casename'] == 'Stochastic_Forcing':
    #     return Stochastic_Frorcing(paramlist)
    else:
        print('case not recognized')
    return



cdef class CasesBase:
    def __init__(self, namelist):
        return
    cpdef initialize_surface(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return

    cpdef initialize_forcing(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return

    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
        return

    cpdef update_surface(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return

    cpdef update_forcing(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return

cdef class HeldSuarez(CasesBase):
    def __init__(self, namelist):
        self.casename = 'HelzSuarez'
        self.tau = 50.0*24.0*3600.0
        self.dTdy        = 60.0
        self.T_lapserate = 10.0
        self.p_ref       = 10e5
        #Temperature profiles from Held and Suarez (1994) -- initialize isothermally in each layer
        self.Tinit  = np.array([229.0, 257.0, 295.0]) # make this a matrix size self.T.values 
        self.Fo  = Forcing.Forcing_HelzSuarez()
        self.Sur = Surface.SurfaceNone()

        return

    cpdef initialize_surface(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        self.Sur.initialize(Gr, PV, DV, namelist)
        return

    cpdef initialize_forcing(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        self.Fo.initialize(Gr, PV, DV, namelist)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        self.Fo.initialize_io(Stats)
        self.Sur.initialize_io(Stats)
        return

    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        CasesBase.io(self, Gr, TS, Stats)
        self.Fo.io(Gr, TS, Stats)
        self.Sur.io(Gr, TS, Stats)
        return

    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
        CasesBase.stats_io(self, TS, Stats)
        self.Fo.stats_io(TS, Stats)
        self.Sur.stats_io(TS, Stats)
        return

    cpdef update_surface(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        self.Sur.update(TS, Gr, PV, DV, namelist)
        return

    cpdef update_forcing(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        self.Fo.update(TS, Gr, PV, DV, namelist)
        return