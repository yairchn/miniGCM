import numpy as np
import matplotlib.pyplot as plt
import sphericalForcing as spf
import time
import scipy as sc
from math import *
import PrognosticVariables
import DiagnosticVariables

class ForcingNone():
	def __init__(self):
		return
	def initialize(self, Pr):
		return
	def update(self, Pr, Gr, TS, PV, DV, namelist):
		return
	def initialize_io(self, Stats):
		return
	def io(self, Pr, TS, Stats):
		return

class ForcingHelzSuarez:
	def __init__(self):
		return

	def initialize(self, Pr):
		# constants from Held & Suarez (1994)
		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		return

	def initialize_io(self, Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	def update(self, Pr, Gr, TS, PV, DV):
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

	def io(self, Pr, TS, Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	def stats_io(self, TS, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		return

class ForcingHelzSuarezMoist:
	def __init__(self):
		return

	def initialize(self, Pr):
		self.Tbar  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_T = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		return

	def initialize_io(self, Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	def update(self, Pr, Gr, TS,  PV, DV):
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

	def io(self, Pr, TS, Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	def stats_io(self, TS, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		return
