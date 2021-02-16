#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from TimeStepping cimport TimeStepping
import sys
from Parameters cimport Parameters
from libc.math cimport pow, log, sin, cos, fmax

cdef extern from "forcing_functions.h":
	void focring_hs(double kappa, double p_ref, double sigma_b, double k_a, double k_b,
			double k_f, double k_s, double Dtheta_z, double T_equator, double DT_y,
			double* p, double* T, double* T_bar, double* sin_lat, double* cos_lat, double* u,
			double* v, double* u_forc, double* v_forc, double* T_forc,
			Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

cdef class ForcingBase:
	def __init__(self):
		return
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.sin_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		self.cos_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		return
	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		return
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		return
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
		return
	cpdef stats_io(self, NetCDFIO_Stats Stats):
		return

cdef class ForcingNone(ForcingBase):
	def __init__(self):
		ForcingBase.__init__(self)
		return
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
		return
	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		return
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		return
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
		return
	cpdef stats_io(self, NetCDFIO_Stats Stats):
		return

cdef class HelzSuarez(ForcingBase):
	def __init__(self):
		ForcingBase.__init__(self)
		return

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
		cdef:
			Py_ssize_t i,j
			Py_ssize_t nx = Pr.nlats
			Py_ssize_t ny = Pr.nlons

		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		cdef:
			Py_ssize_t nx = Pr.nlats
			Py_ssize_t ny = Pr.nlons
			Py_ssize_t nl = Pr.n_layers

		with nogil:
			focring_hs(Pr.kappa, Pr.p_ref, Pr.sigma_b, Pr.k_a, Pr.k_b, Pr.k_f, Pr.k_s,
						Pr.Dtheta_z, Pr.T_equator, Pr.DT_y, &PV.P.values[0,0,0],
						&PV.T.values[0,0,0],&self.Tbar[0,0,0], &self.sin_lat[0,0],
						&self.cos_lat[0,0], &DV.U.values[0,0,0], &DV.V.values[0,0,0],
						&DV.U.forcing[0,0,0], &DV.V.forcing[0,0,0], &PV.T.forcing[0,0,0],
						nx, ny, nl)
		return

	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, NetCDFIO_Stats Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar)
		return