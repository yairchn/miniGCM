import cython
from Grid cimport Grid
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from Parameters cimport Parameters
import os
import netCDF4 as nc
import sphericalForcing as spf
from TimeStepping cimport TimeStepping

cdef class Restart:
    cdef:
        double [:,:] Temperature_2D
        double [:,:] Divergence_2D
        double [:,:] Vorticity_2D
        double [:,:] Pressure_2D

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV,  TimeStepping TS, namelist)
