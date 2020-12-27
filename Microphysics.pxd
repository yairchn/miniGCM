import matplotlib.pyplot as plt
import numpy as np
from math import *
import PrognosticVariables
import DiagnosticVariables
import Parameters

cdef class MicrophysicsBase:
    cdef:
        double [:,:,:] dQTdt
        double [:,:,:] dTdt
        double [:,:] RainRate
    cpdef initialize(self, Parameters Pr, namelist)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class MicrophysicsNone(MicrophysicsBase):
    cdef:
        double [:,:,:] dQTdt
        double [:,:,:] dTdt
        double [:,:] RainRate

    cpdef initialize(self, Parameters Pr, namelist)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)


cdef class MicrophysicsCutoff(MicrophysicsBase):
    cdef:
        double [:,:,:] QL
        double [:,:,:] QV
        double [:,:,:] QR
        double [:,:,:] dQTdt
        double [:,:,:] dTdt
        double [:,:] RainRate

    cpdef initialize(self, Parameters Pr, namelist)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)