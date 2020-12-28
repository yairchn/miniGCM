import cython
from Grid cimport Grid
from math import *
from DiagnosticVariables cimport DiagnosticVariables
import matplotlib.pyplot as plt
import netCDF4
from NetCDFIO cimport NetCDFIO_Stats
import numpy as np
import scipy as sc
import sys
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

cdef class PrognosticVariable:
    cdef:
        double [:,:,:] values
        double [:,:,:] VerticalFlux
        double complex [:,:] spectral
        double complex [:,:] old
        double complex [:,:] now
        double complex [:,:] tendency
        double complex [:,:] sp_VerticalFlux
        double complex [:,:] forcing
        str kind
        str name
        str units

cdef class PrognosticVariables:
    cdef:
        PrognosticVariable Vorticity
        PrognosticVariable Divergence
        PrognosticVariable T
        PrognosticVariable QT
        PrognosticVariable P
        double [:] T_init
        double [:] P_init
        double [:] QT_init
        Py_ssize_t k

    # cpdef initialize(self, Parameters Pr)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr)
    cpdef spectral_to_physical(self, Parameters Pr, Grid Gr)
    cpdef set_old_with_now(self)
    cpdef set_now_with_tendencies(self)
    cpdef reset_pressures(self, Parameters Pr)
    cpdef stats_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)