import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

cdef class DiagnosticVariable:
    cdef:
        double [:,:,:] values
        double [:,:,:] forcing
        double [:,:] SurfaceFlux
        str kind
        str name
        str units

cdef class DiagnosticVariables:
    cdef:
        DiagnosticVariable QL
        DiagnosticVariable Vel
        DiagnosticVariable KE
        DiagnosticVariable P

    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV)