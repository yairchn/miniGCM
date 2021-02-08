import cython
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

cdef class ForcingBase:
	cdef:
		double [:,:,:] Tbar
		double [:,:,:] k_T
		double [:,:,:] k_v
		double [:,:] sin_lat
		double [:,:] cos_lat
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class ForcingNone(ForcingBase):
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class HelzSuarez(ForcingBase):
	cdef:
		Py_ssize_t i
		Py_ssize_t j
		Py_ssize_t k
		Py_ssize_t nx
		Py_ssize_t ny
		Py_ssize_t nl
		Py_ssize_t nlm
		double P_half
		double sigma_ratio

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class HelzSuarezMoist(ForcingBase):
	cdef:
		Py_ssize_t k
		double sigma_b
		double k_a
		double k_s
		double k_f
		double DT_y
		double Dtheta_z
		double cp
		double Rd
		double kappa
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)