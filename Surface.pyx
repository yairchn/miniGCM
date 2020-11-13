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
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class SurfaceNone():
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
        return