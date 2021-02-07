import cython
from Grid cimport Grid
import numpy as np
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
import pylab as plt

cdef extern from "forcing_functions.h":
	void focring_hs(double kappa, double p_ref, double sigma_b, double k_a, double k_b,
			double k_f, double k_s, double Dtheta_z, double T_equator, double DT_y,
			double* p, double* T, double* T_bar, double* lat, double* u,
			double* v, double* u_forc, double* v_forc, double* T_forc,
			Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil
	void print_c(double* T, double* var, double*  var_2d,
		         Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil


cdef class ForcingBase:
	def __init__(self):
		return
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
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
		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		# self.sin_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		# self.cos_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		self.sin_lat = np.sin(Gr.lat)
		self.cos_lat = np.cos(Gr.lat)
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
			double P_half, sigma_ratio

		with nogil:
			# focring_hs(Pr.kappa, Pr.p_ref, Pr.sigma_b, Pr.k_a, Pr.k_b, Pr.k_f, Pr.k_s,
			# 			Pr.Dtheta_z, Pr.T_equator, Pr.DT_y, &PV.P.values[0,0,0], &PV.T.values[0,0,0],
			# 			&self.Tbar[0,0,0], &Gr.lat[0,0], &DV.U.values[0,0,0], &DV.V.values[0,0,0],
			# 			&DV.U.forcing[0,0,0], &DV.V.forcing[0,0,0], &PV.T.forcing[0,0,0], nx, ny, nl)

			for i in range(nx):
				for j in range(ny):
					for k in range(nl):
						P_half = (PV.P.values[i,j,k]+PV.P.values[i,j,k+1])/2.0
						self.Tbar[i,j,k] = fmax(
										   (Pr.T_equator - Pr.DT_y*self.sin_lat[i,j]*self.sin_lat[i,j] -
							                Pr.Dtheta_z*log(P_half/Pr.p_ref)*self.cos_lat[i,j]*self.cos_lat[i,j])*
						                    pow(P_half/Pr.p_ref , Pr.kappa)
						                    ,200.0)

						sigma_ratio = fmax((P_half/PV.P.values[i,j,nl]-Pr.sigma_b)/(1-Pr.sigma_b),0.0)
						self.k_T[i,j,k] = Pr.k_a + (Pr.k_s-Pr.k_a)*sigma_ratio*pow(self.cos_lat[i,j],4.0)
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

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.sin_lat = np.sin(Gr.lat)
		self.cos_lat = np.cos(Gr.lat)
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	@cython.wraparound(False)
	@cython.boundscheck(False)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		cdef:
			Py_ssize_t i,j,k, ijk
			Py_ssize_t nx = Pr.nlats
			Py_ssize_t ny = Pr.nlons
			Py_ssize_t nl = Pr.n_layers
			Py_ssize_t I = 4
			Py_ssize_t J = 5
			Py_ssize_t K = 3
			Py_ssize_t nlm = Gr.SphericalGrid.nlm
			double P_half, sigma_ratio
			double [:,:,:] Tbar_c  = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
			double [:,:] var_2d = np.zeros((nx,ny),dtype=np.float64, order='c')



		with nogil:
			for i in range(nx):
				for j in range(ny):
					for k in range(nl):
						P_half = (PV.P.values[i,j,k]+PV.P.values[i,j,k+1])/2.0
						self.Tbar[i,j,k] = fmax(
										   (Pr.T_equator - Pr.DT_y*self.sin_lat[i,j]*self.sin_lat[i,j] -
							                Pr.Dtheta_z*log(P_half/Pr.p_ref)*self.cos_lat[i,j]*self.cos_lat[i,j])*
						                    pow(P_half/Pr.p_ref , Pr.kappa)
						                    ,200.0)
						sigma_ratio = fmax((P_half/PV.P.values[i,j,nl]-Pr.sigma_b)/(1-Pr.sigma_b),0.0)
						self.k_T[i,j,k] = Pr.k_a + (Pr.k_s-Pr.k_a)*sigma_ratio*pow(self.cos_lat[i,j],4.0)
						self.k_v[i,j,k] = Pr.k_b + Pr.k_f*sigma_ratio

						DV.U.forcing[i,j,k] = -self.k_v[i,j,k]*DV.U.values[i,j,k]
						DV.V.forcing[i,j,k] = -self.k_v[i,j,k]*DV.V.values[i,j,k]
						PV.T.forcing[i,j,k] = -self.k_T[i,j,k]*(PV.T.values[i,j,k]-self.Tbar[i,j,k])
		with nogil:
			focring_hs(Pr.kappa, Pr.p_ref, Pr.sigma_b, Pr.k_a, Pr.k_b, Pr.k_f, Pr.k_s,
						Pr.Dtheta_z, Pr.T_equator, Pr.DT_y, &PV.P.values[0,0,0], &PV.T.values[0,0,0],
						&Tbar_c[0,0,0], &Gr.lat[0,0], &DV.U.values[0,0,0], &DV.V.values[0,0,0],
						&DV.U.forcing[0,0,0], &DV.V.forcing[0,0,0], &PV.T.forcing[0,0,0], nx, ny, nl)
		if np.max(np.abs(np.subtract(self.Tbar,Tbar_c)))>1e-10:
			plt.figure('Tbar')
			plt.contourf(np.subtract(self.Tbar,Tbar_c))
			plt.colorbar()
			plt.show()
		return

	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, NetCDFIO_Stats Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar)
		return
