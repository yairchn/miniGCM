import matplotlib.pyplot as plt
import numpy as np
from math import *
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport PrognosticVariables
from PrognosticVariables cimport PrognosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats

cdef class MicrophysicsBase:
    cdef:
        double [:,:] RainRate
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, namelist)
    cpdef update(self, Parameters Pr, TimeStepping TS, PrognosticVariables PV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class MicrophysicsNone(MicrophysicsBase):
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, namelist)
    cpdef update(self, Parameters Pr, TimeStepping TS, PrognosticVariables PV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)


cdef class MicrophysicsCutoff(MicrophysicsBase):
    cdef:
        double [:,:,:] QL
        double [:,:,:] QV
        double [:,:,:] QR
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, namelist)
    cpdef update(self, Parameters Pr, TimeStepping TS, PrognosticVariables PV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats)
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)