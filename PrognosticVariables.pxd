import cython
from concurrent.futures import ThreadPoolExecutor
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
import Microphysics
cimport Microphysics
from NetCDFIO cimport NetCDFIO_Stats
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping

cdef class PrognosticVariable:
    cdef:
        double [:,:,:] values
        double [:,:,:] HyperDiffusion
        double [:,:,:] VerticalFlux
        double [:,:] SurfaceFlux
        double [:,:,:] forcing
        double [:,:,:] mp_tendency
        double [:,:,:] old
        double [:,:,:] now
        double [:,:,:] tendency
        double [:,:,:] ZonalFlux
        double [:,:,:] MeridionalFlux
        str kind
        str name
        str units
    cpdef set_bcs(self, Parameters Pr, Grid Gr)

cdef class PrognosticVariables:
    cdef:
        Py_ssize_t k
        PrognosticVariable U
        PrognosticVariable V
        PrognosticVariable H
        PrognosticVariable QT
        double [:] H_init
        double [:] QT_init
        double [:] amp_dHdp
        object MP


    cpdef initialize(self, Parameters Pr)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef apply_bc(self, Parameters Pr, Grid Gr)
    cpdef set_old_with_now(self)
    cpdef set_now_with_tendencies(self)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
