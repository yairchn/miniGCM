import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
from DiagnosticVariables cimport DiagnosticVariables
from math import *
import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
import sphericalForcing as spf
import time
from TimeStepping cimport TimeStepping
import sys
from Parameters cimport Parameters

cdef class SurfaceBase:
    cdef:
        double [:,:] U_flux
        double [:,:] V_flux
        double [:,:] T_flux
        double [:,:] QT_flux
        double [:,:] T_surf
        double [:,:] U_abs
        double [:,:] QT_surf

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)


cdef class SurfaceNone(SurfaceBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class SurfaceBulkFormula(SurfaceBase):
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)