import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import NetCDFIO as NetCDFIO_Stats
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
from TimeStepping cimport TimeStepping
import numpy as np
import time
import sys

cdef class DiagnosticVariable:
    cdef:
        double [:,:,:] values
        double [:,:] spectral
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
        Py_ssize_t k
        # Py_ssize_t nlats
        # Py_ssize_t nlons
        # Py_ssize_t n_layers

    cpdef initialize(self, Grid Gr)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef physical_to_spectral(self, Grid Gr)
    # cpdef spectral_to_physical(self)
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
    cpdef update(self, Grid Gr, PrognosticVariables PV)