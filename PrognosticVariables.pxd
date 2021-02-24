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
        double [:,:,:] VerticalFlux
        double [:,:] SurfaceFlux
        double [:,:,:] forcing
        double [:,:,:] mp_tendency
        double [:,:,:] old
        double [:,:,:] now
        double [:,:,:] tendency
        double [:,:,:] Weno_dFdx
        double [:,:,:] Weno_dFdy
        str kind
        str name
        str units
    cpdef set_bcs(self, Parameters Pr, Grid Gr)

cdef class PrognosticVariables:
    cdef:
        Py_ssize_t k
        PrognosticVariable U
        PrognosticVariable V
        PrognosticVariable T
        PrognosticVariable QT
        PrognosticVariable P
        double [:] T_init
        double [:] P_init
        double [:] QT_init
        double [:] amp_dTdp
        object MP


    cpdef initialize(self, Parameters Pr)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef apply_bc(self, Parameters Pr, Grid Gr)
    cpdef set_old_with_now(self)
    cpdef set_now_with_tendencies(self)
    cpdef reset_pressures_and_bcs(self, Parameters Pr, DiagnosticVariables DV)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
