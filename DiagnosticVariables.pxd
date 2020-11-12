import numpy as np
import matplotlib.pyplot as plt
import time
# import sphericalForcing as spf
import NetCDFIO as Stats
from math import *

cdef class DiagnosticVariable:
    cdef:
        double [:,:,:] values
        double [:,:,:] spectral
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

    cpdef initialize(self, Grid Gr)
    cpdef initialize_io(self, NetCDFIO Stats)
    cpdef physical_to_spectral(self, Grid Gr)
    cpdef spectral_to_physical(self)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO Stats)
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO Stats)
    cpdef update(self, Grid Gr, PrognosticVariables PV)