import matplotlib.pyplot as plt
import scipy as sc
import netCDF4
import numpy as np
from math import *

cdef class PrognosticVariable:
    cdef:
        double [:,:,:] values
        double [:,:,:] spectral
        double [:,:,:] old
        double [:,:,:] now
        double [:,:,:] tendency
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
        double [:,:,:] Base_pressure
        double [:,:,:] T_init
        double [:,:,:] P_init
        double [:,:,:] QT_init

    cpdef initialize(self, Grid Gr, DiagnosticVariable DV)
    cpdef initialize_io(self, NetCDFIO Stats)
    cpdef physical_to_spectral(self, Grid Gr)
    cpdef spectral_to_physical(self, Grid Gr)
    cpdef set_old_with_now(self)
    cpdef set_now_with_tendencies(self)
    cpdef reset_pressures(self, Grid Gr)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO Stats)
    cpdef io(self, Gr, TimeStepping TS, NetCDFIO Stats)
    cpdef compute_tendencies(self, Grid Gr, PrognosticVariables, DiagnosticVariable DV, namelist)