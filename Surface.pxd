import cython
from Grid cimport Grid
from DiagnosticVariables cimport DiagnosticVariables
from math import *
import matplotlib.pyplot as plt
import numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
import sphericalForcing as spf
import time
from TimeStepping cimport TimeStepping
import sys


cdef class SurfaceBase:
    cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)


cdef class SurfaceNone():
    cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)