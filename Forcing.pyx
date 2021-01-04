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
		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		cdef:
			Py_ssize_t k
			double [:,:] P_half
			double [:,:] pressure_ratio
			double [:,:] Ty
			double [:,:] Tp
			double [:,:] exner

		for k in range(Pr.n_layers):
			P_half = np.divide(np.add(PV.P.values[:,:,k],PV.P.values[:,:,k+1]),2.0)
			pressure_ratio = np.divide(P_half,Pr.p_ref)
			exner  = np.power(pressure_ratio,Pr.kappa)
			Ty = np.multiply(Pr.DT_y,np.power(np.sin(Gr.lat),2))
			Tp = np.multiply(Pr.Dtheta_z,np.log(pressure_ratio)*np.power(np.cos(Gr.lat),2))
			self.Tbar.base[:,:,k] = np.clip(np.multiply(np.subtract(np.subtract(Pr.T_equator,Ty),Tp),exner), 200.0, None)

			sigma = np.divide(P_half,PV.P.values[:,:,Pr.n_layers])
			sigma_ratio = np.clip(np.divide(sigma-Pr.sigma_b,1-Pr.sigma_b),0,None)
			self.k_T.base[:,:,k] = np.add(Pr.k_a, np.multiply((Pr.k_s-Pr.k_a),np.multiply(sigma_ratio,np.power(np.cos(Gr.lat),4))))
			self.k_v.base[:,:,k] = np.add(Pr.k_a, Pr.k_f*sigma_ratio)

			PV.Vorticity.forcing.base[:,k] ,PV.Divergence.forcing.base[:,k] = (
              	Gr.SphericalGrid.getvrtdivspec(-np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k]),
											   -np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])))
			PV.T.forcing.base[:,k] = -Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],np.subtract(PV.T.values[:,:,k],self.Tbar[:,:,k])))
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
		Pr.sigma_b = namelist['forcing']['sigma_b']
		Pr.k_a = namelist['forcing']['k_a']
		Pr.k_s = namelist['forcing']['k_s']
		Pr.k_f = namelist['forcing']['k_f']
		Pr.DT_y = namelist['forcing']['equator_to_pole_dT']
		Pr.T_equator = namelist['forcing']['equatorial_temperature']
		Pr.Dtheta_z = namelist['forcing']['lapse_rate']
		Pr.Tbar0 = namelist['forcing']['relaxation_temperature']

		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
		cdef:
			Py_ssize_t k
			double [:,:] P_half
			double [:,:] pressure_ratio

		for k in range(Pr.n_layers):
			P_half = np.divide(np.add(PV.P.values[:,:,k],PV.P.values[:,:,k+1]),2.0)
			pressure_ratio = np.divide(P_half,Pr.p_ref)
			self.Tbar.base[:,:,k] = np.clip((Pr.T_equator-Pr.DT_y*np.power(np.sin(Gr.lat),2)-Pr.Dtheta_z*np.log(pressure_ratio)
				               *np.power(np.cos(Gr.lat),2))*np.power(pressure_ratio,Pr.kappa), 200.0, None)
			sigma=np.divide(P_half,PV.P.values[:,:,Pr.n_layers])
			sigma_ratio=np.clip(np.divide(sigma-Pr.sigma_b,1-Pr.sigma_b),0,None)
			self.k_T.base[:,:,k] = Pr.k_a+(Pr.k_s-Pr.k_a)*np.multiply(sigma_ratio,np.power(np.cos(Gr.lat),4))
			self.k_v.base[:,:,k] = Pr.k_a+Pr.k_f*sigma_ratio
			PV.Vorticity.forcing.base[:,k] ,PV.Divergence.forcing.base[:,k] = (
              	Gr.SphericalGrid.getvrtdivspec(-np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k]),
											   -np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])))
			PV.T.forcing.base[:,k] = -Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],np.subtract(PV.T.values[:,:,k],self.Tbar[:,:,k])))
		return

	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, NetCDFIO_Stats Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar)
		return
