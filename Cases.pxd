import cython
import numpy as np
cimport numpy as np
from concurrent.futures import ThreadPoolExecutor
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from Forcing cimport ForcingBase
from Surface cimport SurfaceBase
from Microphysics cimport MicrophysicsBase
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
    cpdef initialize_forcing(self, Parameters Pr, Grid Gr, namelist)
    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV,namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)

cdef class DryVortex(CaseBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_forcing(self, Parameters Pr, Grid Gr, namelist)
    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV,namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)

cdef class MoistVortex(CaseBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_forcing(self, Parameters Pr, Grid Gr, namelist)
    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV,namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)

cdef class ReedJablonowski(CaseBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef initialize_forcing(self, Parameters Pr, Grid Gr, namelist)
    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV,namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)