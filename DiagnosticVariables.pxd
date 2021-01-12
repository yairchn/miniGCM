import cython
import numpy as np
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

cdef class DiagnosticVariable:
    cdef:
        double [:,:,:] values
        double complex [:,:] spectral
        str kind
        str name
        str units

cdef class DiagnosticVariables:
    cdef:
        DiagnosticVariable QL
        DiagnosticVariable U
        DiagnosticVariable V
        DiagnosticVariable KE
        DiagnosticVariable gZ
        DiagnosticVariable Wp
        Py_ssize_t j, k

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV)