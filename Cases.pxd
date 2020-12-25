import numpy as np
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from Forcing cimport ForcingBase
from Surface cimport SurfaceBase
import cython
import sys
from TimeStepping cimport TimeStepping


cdef class CasesBase:
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

    cpdef initialize_surface(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef initialize_forcing(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef update_surface(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef update_forcing(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)

cdef class HeldSuarez(CasesBase):
    cpdef initialize_surface(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef initialize_forcing(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef update_surface(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef update_forcing(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)