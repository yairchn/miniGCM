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
	def initialize(self, Gr, PV, DV, namelist):
                self.Tbar                       = np.zeros((Gr.nx,Gr.ny,Gr.nz), dtype=np.double, order='c')
                self.Tbar_meridional            = np.zeros((      Gr.ny,Gr.nz), dtype=np.double, order='c')
                self.pressure_ratio_meridional  = np.zeros(       Gr.ny,        dtype=np.double, order='c')
                return
	def update(self, TS, Gr, PV, DV, namelist):
		return
	def initialize_io(self, Stats):
		return
	def io(self, Stats):
		return

class Forcing_HelzSuarez:
	def __init__(self):
		return

	def initialize(self, Gr, PV, DV, namelist):
		# constants from Held & Suarez (1994)
		self.PV = PrognosticVariables.PrognosticVariables(Gr)
		self.DV = DiagnosticVariables.DiagnosticVariables(Gr)
		self.Tbar  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.sigma_b = namelist['forcing']['sigma_b']
		self.k_a = namelist['forcing']['k_a']
		self.k_b = namelist['forcing']['k_b']
		self.k_s = namelist['forcing']['k_s']
		self.k_f = namelist['forcing']['k_f']
		self.DT_y = namelist['forcing']['DT_y']
		self.Dtheta_z = namelist['forcing']['lapse_rate']
		self.Tbar0 = namelist['forcing']['relaxation_temperature']
		self.cp = namelist['thermodynamics']['heat_capacity']
		self.Rd = namelist['thermodynamics']['ideal_gas_constant']
		self.kappa = self.Rd/self.cp
		self.k_T = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')

		return

	def initialize_io(self, Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	def update(self, TS, Gr, PV, DV, namelist):
		for k in range(Gr.n_layers):
			self.pressure_ratio=(PV.P.values[:,:,k]+PV.P.values[:,:,k+1])/(2.0*Gr.p_ref)
			self.Tbar[:,:,k] = ((315.0-self.DT_y*np.sin(Gr.lat)**2-self.Dtheta_z*np.log(self.pressure_ratio)*np.cos(Gr.lat)**2)*(self.pressure_ratio)**Gr.kappa)
			self.Tbar[:,:,k][self.Tbar[:,:,k]<=200.0]=200.0
			sigma=np.divide((PV.P.values[:,:,k]+PV.P.values[:,:,k+1])/2.,PV.P.values[:,:,Gr.n_layers])
			sigma_ratio=np.clip(np.divide(sigma-self.sigma_b,1-self.sigma_b),0,None)
			self.k_T[:,:,k] = self.k_a+(self.k_s-self.k_a)*np.multiply(sigma_ratio,np.power(np.cos(Gr.lat),4))
			self.k_v[:,:,k] = self.k_b+self.k_f*sigma_ratio
			PV.Vorticity.forcing[:,k] ,PV.Divergence.forcing[:,k] = (
              	Gr.SphericalGrid.getvrtdivspec(-np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k]),
												-np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])))
			PV.T.forcing[:,k] = -Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],(PV.T.values[:,:,k]-self.Tbar[:,:,k])))
			#plt.figure(k)
			#plt.contourf(self.Tbar[:,:,k])
			#plt.colorbar()
#
		#plt.show()
		return

	def io(self, Gr, TS, Stats):
		Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'T_eq', self.Tbar)
		return

	def stats_io(self, TS, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		return
