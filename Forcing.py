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

cdef class ForcingBase:
	def __init__(self):
		return
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.QTbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.Divergence_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
		self.Vorticity_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
		self.T_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
		self.QT_forcing = np.zeros((Gr.SphericalGrid.nlm, Pr.n_layers), dtype=np.float64, order='c')
		return
	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		return
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		return
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
		return
	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
		return

cdef class ForcingNone(ForcingBase):
	def __init__(self):
		ForcingBase.__init__(self)
		return
	cpdef initialize(self, PrognosticVariables Pr):
		return
	cpdef update(self, PrognosticVariables Pr, Grid Gr, TimeStepping TS, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		return
	cpdef initialize_io(self, Stats):
		return
	cpdef io(self, PrognosticVariables Pr, TimeStepping TS, Stats):
		return

cdef class ForcingHelzSuarez(ForcingBase):
	def __init__(self):
		ForcingBase.__init__(self)
		return

	cpdef initialize(self, Parameters Pr):
		# constants from Held & Suarez (1994)
		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		return

	cpdef initialize_io(self, Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	cpdef update(self, Parameters Pr, Grid Gr, TimeStepping TS, PV, DV):
		for k in range(Pr.n_layers):
			self.pressure_ratio = (PV.P.values[:,:,k]+PV.P.values[:,:,k+1])/(2.0*Pr.p_ref)
			self.Tbar[:,:,k] = ((Pr.T_equator-Pr.DT_y*np.sin(Gr.lat)**2-Pr.Dtheta_z*np.log(self.pressure_ratio)*np.cos(Gr.lat)**2)*(self.pressure_ratio)**Pr.kappa)
			self.Tbar[:,:,k][self.Tbar[:,:,k]<=200.0]=200.0
			sigma = np.divide((PV.P.values[:,:,k]+PV.P.values[:,:,k+1])/2.,PV.P.values[:,:,Pr.n_layers])
			sigma_ratio = np.clip(np.divide(sigma-Pr.sigma_b,1-Pr.sigma_b),0,None)
			self.k_T[:,:,k] = Pr.k_a+(Pr.k_s-Pr.k_a)*np.multiply(sigma_ratio,np.power(np.cos(Gr.lat),4))
			self.k_v[:,:,k] = Pr.k_a+Pr.k_f*sigma_ratio
			PV.Vorticity.forcing[:,k] ,PV.Divergence.forcing[:,k] = (
              	Gr.SphericalGrid.getvrtdivspec(-np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k]),
												-np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])))
			PV.T.forcing[:,k] = -Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],(PV.T.values[:,:,k]-self.Tbar[:,:,k])))
		return

	cpdef io(self, Parameters Pr, TimeStepping TS, Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, TimeStepping TS, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		return

cdef class ForcingHelzSuarezMoist(ForcingBase):
	def __init__(self):
		ForcingBase.__init__(self)
		return

	cpdef initialize(self, Parameters Pr):
		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		return

	cpdef initialize_io(self, Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	cpdef update(self, Parameters Pr, Grid Gr, TimeStepping TS, PrognosticVariables PV, DiagnosticVariables DV):
		for k in range(Pr.n_layers):
			self.pressure_ratio=(PV.P.values[:,:,k]+PV.P.values[:,:,k+1])/(2.0*Pr.p_ref)
			self.Tbar[:,:,k]=((Pr.T_equator-Pr.DT_y*np.sin(Gr.lat)**2-Pr.Dtheta_z*np.log(self.pressure_ratio)*np.cos(Gr.lat)**2)*(self.pressure_ratio)**Pr.kappa)
			self.Tbar[:,:,k][self.Tbar[:,:,k]<=200.0]=200.0
			sigma=np.divide((PV.P.values[:,:,k]+PV.P.values[:,:,k+1])/2.,PV.P.values[:,:,Pr.n_layers])
			sigma_ratio=np.clip(np.divide(sigma-Pr.sigma_b,1-Pr.sigma_b),0,None)
			self.k_T[:,:,k] = Pr.k_a+(Pr.k_s-Pr.k_a)*np.multiply(sigma_ratio,np.power(np.cos(Gr.lat),4))
			self.k_v[:,:,k] = Pr.k_a+Pr.k_f*sigma_ratio
			PV.Vorticity.forcing[:,k] ,PV.Divergence.forcing[:,k] = (
              	Gr.SphericalGrid.getvrtdivspec(-np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k]),
												-np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])))
			PV.T.forcing[:,k] = -Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],(PV.T.values[:,:,k]-self.Tbar[:,:,k])))
		return

	cpdef io(self, Parameters Pr, TimeStepping TS, Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	cpdef stats_io(self, TimeStepping TS, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		return
