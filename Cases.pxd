import numpy as np
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from Forcing cimport ForcingBase
from Surface cimport SurfaceBase
from Microphysics cimport MicrophysicsBase
import cython
import sys
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters


cdef class CaseBase:
    cdef:
        str casename
        double tau
        double dTdy
        double T_lapserate
        double p_ref
        double foring_type
        double [:] Tinit
        SurfaceBase Sur
        ForcingBase Fo
        MicrophysicsBase MP

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_forcing(self, Parameters Pr, namelist)
    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef update_forcing(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef update_microphysics(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TimeStepping TS)

cdef class HeldSuarez(CaseBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_forcing(self, Parameters Pr, namelist)
    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef update_forcing(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef update_microphysics(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TimeStepping TS)

cdef class HeldSuarezMoist(CaseBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_forcing(self, Parameters Pr, namelist)
    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef update_forcing(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef update_microphysics(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TimeStepping TS)