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

cdef class ForcingBase:
	cdef:
		double [:,:,:] Tbar
		double [:,:,:] QTbar
		double [:,:,:] Divergence_forcing
		double [:,:,:] Vorticity_forcing
		double [:,:,:] T_forcing
		double [:,:,:] QT_forcing
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class ForcingNone(ForcingBase):
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class Forcing_HelzSuarez(ForcingBase):
	cdef:
		double [:,:,:] k_T
		double [:,:,:] k_Q
		double [:,:,:] k_v
		double sigma_b
		double k_a
		double k_s
		double k_f
		double DT_y
		double Dtheta_z
		double cp
		double Rd
		double kappa
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)

cdef class Forcing_HelzSuarez_moist(ForcingBase):
	cdef:
		double [:,:,:] k_T
		double [:,:,:] k_Q
		double [:,:,:] k_v
		double sigma_b
		double k_a
		double k_s
		double k_f
		double DT_y
		double Dtheta_z
		double cp
		double Rd
		double kappa
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats)