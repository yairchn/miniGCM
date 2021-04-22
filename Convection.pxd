#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

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
import sphericalForcing as spf

cdef class ConvectionBase:
    cdef:
        double complex [:] F0
        double complex [:] sph_noise
        double complex [:,:] Wp
        double complex [:,:] wT
        double complex [:,:] wQT
        double complex [:,:] wVort
        double complex [:,:] wDiv

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class ConvectionNone(ConvectionBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class ConvectionRandom(ConvectionBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
