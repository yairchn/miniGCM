import cython
from Grid cimport Grid
import numpy as np
import matplotlib.pyplot as plt
import sphericalForcing as spf
import time
import scipy as sc
from math import *
from NetCDFIO cimport NetCDFIO_Stats
import scipy as sc
from TimeStepping cimport TimeStepping
import sys
from Parameters cimport Parameters
from libc.math cimport pow, log, sin, cos, fmax

cdef class ForcingBase:
	def __init__(self):
		return
	cpdef initialize(self, Parameters Pr, namelist):
		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.QTbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.Divergence_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
		self.Vorticity_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
		self.T_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
		self.QT_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
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
	cpdef initialize(self, Parameters Pr, namelist):
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

	cpdef initialize(self, Parameters Pr, namelist):
		# constants from Held & Suarez (1994)
		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	@cython.wraparound(False)
	@cython.boundscheck(False)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		cdef:
			Py_ssize_t i,j,k
			Py_ssize_t nx = Pr.nlats
			Py_ssize_t ny = Pr.nlons
			Py_ssize_t nl = Pr.n_layers
			Py_ssize_t nlm = Gr.SphericalGrid.nlm
			double P_half, pressure_ratio, exner, sigma_ratio,sigma, Ty, Tp
			# double [:,:] Ty = np.zeros((nx, ny),dtype=np.float64, order='c')
			# double [:,:] Tp = np.zeros((nx, ny),dtype=np.float64, order='c')

		with nogil:
			for i in range(nx):
				for j in range(ny):
					for k in range(nl):
						P_half = (PV.P.values[i,j,k]+PV.P.values[i,j,k+1])/2.0
						pressure_ratio = (P_half/Pr.p_ref)
						exner = pow(pressure_ratio,Pr.kappa)
						Ty = Pr.DT_y*pow(sin(Gr.lat[i,j]),2.0)
						Tp = Pr.Dtheta_z*log(pressure_ratio)*pow(cos(Gr.lat[i,j]),2.0)
						self.Tbar[i,j,k] = fmax((Pr.T_equator-Ty-Tp)*exner, 200.0)

						sigma = (P_half/PV.P.values[i,j,nl])
						sigma_ratio = fmax((sigma-Pr.sigma_b)/(1-Pr.sigma_b),0.0)
						self.k_T[i,j,k] = Pr.k_a + (Pr.k_s-Pr.k_a)*sigma_ratio*pow(cos(Gr.lat[i,j]),4.0)
						self.k_v[i,j,k] = Pr.k_b + Pr.k_f*sigma_ratio

						DV.U.forcing[i,j,k] = -self.k_v[i,j,k]*DV.U.values[i,j,k]
						DV.V.forcing[i,j,k] = -self.k_v[i,j,k]*DV.V.values[i,j,k]
						PV.T.forcing[i,j,k] = -self.k_T[i,j,k]*(PV.T.values[i,j,k]-self.Tbar[i,j,k])
		return

	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, NetCDFIO_Stats Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar)
		return

cdef class HelzSuarezMoist(ForcingBase):
	def __init__(self):
		ForcingBase.__init__(self)
		return

	cpdef initialize(self, Parameters Pr, namelist):

		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	@cython.wraparound(False)
	@cython.boundscheck(False)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		cdef:
			Py_ssize_t i,j,k
			Py_ssize_t nx = Pr.nlats
			Py_ssize_t ny = Pr.nlons
			Py_ssize_t nl = Pr.n_layers
			Py_ssize_t nlm = Gr.SphericalGrid.nlm
			double P_half, pressure_ratio, exner, sigma_ratio,sigma, Ty, Tp
			# double [:,:] Ty = np.zeros((nx, ny),dtype=np.float64, order='c')
			# double [:,:] Tp = np.zeros((nx, ny),dtype=np.float64, order='c')

		with nogil:
			for i in range(nx):
				for j in range(ny):
					for k in range(nl):
						P_half = (PV.P.values[i,j,k]+PV.P.values[i,j,k+1])/2.0
						pressure_ratio = (P_half/Pr.p_ref)
						exner = pow(pressure_ratio,Pr.kappa)
						Ty = Pr.DT_y*pow(sin(Gr.lat[i,j]),2.0)
						Tp = Pr.Dtheta_z*log(pressure_ratio)*pow(cos(Gr.lat[i,j]),2.0)
						self.Tbar[i,j,k] = fmax((Pr.T_equator-Ty-Tp)*exner, 200.0)

						sigma = (P_half/PV.P.values[i,j,nl])
						sigma_ratio = fmax((sigma-Pr.sigma_b)/(1-Pr.sigma_b),0.0)
						self.k_T[i,j,k] = Pr.k_a + (Pr.k_s-Pr.k_a)*sigma_ratio*pow(cos(Gr.lat[i,j]),4.0)
						self.k_v[i,j,k] = Pr.k_b + Pr.k_f*sigma_ratio

						DV.U.forcing[i,j,k] = -self.k_v[i,j,k]*DV.U.values[i,j,k]
						DV.V.forcing[i,j,k] = -self.k_v[i,j,k]*DV.V.values[i,j,k]
						PV.T.forcing[i,j,k] = -self.k_T[i,j,k]*(PV.T.values[i,j,k]-self.Tbar[i,j,k])
		return

	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, NetCDFIO_Stats Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar)
		return
