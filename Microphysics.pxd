import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats

cdef class MicrophysicsBase:
    cdef:
        double [:,:] RainRate
        double [:,:,:] qv_star
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class MicrophysicsNone(MicrophysicsBase):
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)


cdef class MicrophysicsCutoff(MicrophysicsBase):
    cdef:
        Py_ssize_t k
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)