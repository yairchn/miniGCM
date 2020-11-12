
import numpy as np
import matplotlib.pyplot as plt
import sphericalForcing as spf
import time
import scipy as sc
from math import *
import PrognosticVariables
import DiagnosticVariables

cdef class ForcingNone:
	cdef:
		double [:,:,:] Tbar
		double [:,:,:] QTbar
		double [:,:,:] Divergence_forcing
		double [:,:,:] Vorticity_forcing
		double [:,:,:] T_forcing
		double [:,:,:] QT_forcing
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef initialize_io(self, NetCDFIO Stats)
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO Stats)
	cpdef stats_io(self, TimeStepping TS, NetCDFIO Stats)

cdef class Forcing_HelzSuarez:
	cdef:
		double [:,:,] Tbar
		double [:,:,] k_T
		double [:,:,] k_Q
		double [:,:,] k_v
		double [:,:,] QTbar
		double [:,:,] sigma_b
		double [:,:,] k_a
		double [:,:,] k_s
		double [:,:,] k_f
		double [:,:,] DT_y
		double [:,:,] Dtheta_z
		double [:,:,] Tbar0
		double [:,:,] cp
		double [:,:,] Rd
		double [:,:,] kappa
		double [:,:,] k_T
		double [:,:,] k_v
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef initialize_io(self, NetCDFIO Stats)
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO Stats)
	cpdef stats_io(self, TimeStepping TS, NetCDFIO Stats)

cdef class Forcing_HelzSuarez_moist:
	cdef:
		double [:,:,:] Tbar
		double [:,:,:] QTbar
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef initialize_io(self, NetCDFIO Stats)
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist)
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO Stats)
	cpdef stats_io(self, TimeStepping TS, NetCDFIO Stats)