import cython
import numpy as np
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
from TimeStepping cimport TimeStepping

cdef class DiagnosticVariable:
    cdef:
        double [:,:,:] values
        double complex [:,:] spectral
        str kind
        str name
        str units

cdef class DiagnosticVariables:
    cdef:
        DiagnosticVariable U
        DiagnosticVariable V
        DiagnosticVariable KE
        DiagnosticVariable gZ
        DiagnosticVariable Wp
        Py_ssize_t k

    cpdef initialize(self, Grid Gr)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef physical_to_spectral(self, Grid Gr)
    # cpdef spectral_to_physical(self)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef update(self, Grid Gr, PrognosticVariables PV)