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
	void focring_bm(double kappa, double p_ref, double tau,
			double* p, double* T, double* T_bar, double* u,
			double* v, double* u_forc, double* v_forc, double* T_forc,
			Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

cdef class ForcingBase:
	def __init__(self):
		return
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
		self.Tbar = np.zeros((Gr.nx, Gr.ny, Gr.nl), dtype=np.float64, order='c')
		return
	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		return
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		return
	cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
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
	cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
		return
	cpdef stats_io(self, NetCDFIO_Stats Stats):
		return

cdef class ForcingBettsMiller(ForcingBase):
	def __init__(self):
		ForcingBase.__init__(self)
		return

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
		cdef:
			Py_ssize_t i,j
			Py_ssize_t nx = Pr.nx
			Py_ssize_t ny = Pr.ny
		self.Tbar = np.zeros((Gr.nx, Gr.ny, Gr.nl), dtype=np.float64, order='c')
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_axisymmetric_mean('axisymmetric_mean_T_eq')
		return

	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		cdef:
			Py_ssize_t nx = Gr.nx
			Py_ssize_t ny = Gr.ny
			Py_ssize_t nl = Gr.nl
			double [:,:] T_surf = np.zeros((Gr.nx, Gr.ny), dtype=np.float64, order='c')

		with nogil:
			focring_bm(Pr.kappa, Pr.p_ref, Pr.tau, &PV.T.values[0,0,0],
						&PV.T.values[0,0,0], &self.Tbar[0,0,0],
						&PV.U.values[0,0,0], &PV.V.values[0,0,0],
						&PV.U.forcing[0,0,0], &PV.V.forcing[0,0,0], &PV.T.forcing[0,0,0],
						nx, ny, nl)
		return

	cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Pr, Gr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, NetCDFIO_Stats Stats):
		Stats.write_axisymmetric_mean('axisymmetric_mean_T_eq', self.Tbar)
		return