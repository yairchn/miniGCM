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
	def __init__(self):
		return
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		self.Tbar = np.zeros((Gr.nx,Gr.ny, Gr.nz), dtype=np.double, order='c')
		self.QTbar = np.zeros((Gr.nx,Gr.ny, Gr.nz), dtype=np.double, order='c')
		self.Divergence_forcing = np.zeros((n_spec,nl), dtype=np.double, order='c')
		self.Vorticity_forcing = np.zeros((n_spec,nl), dtype=np.double, order='c')
		self.T_forcing = np.zeros((n_spec,nl), dtype=np.double, order='c')
		self.QT_forcing = np.zeros((n_spec,nl), dtype=np.double, order='c')
		return
	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		return
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		return
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
		return
	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
		return


cdef class ForcingNone():
	def __init__(self):
		return
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		return
	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		return
	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		return
	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
		return
	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
		return

cdef class Forcing_HelzSuarez:
	def __init__(self):
		return

	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		# constants from Held & Suarez (1994)
		self.PV = PrognosticVariables.PrognosticVariables(Gr)
		self.DV = DiagnosticVariables.DiagnosticVariables(Gr)
		self.Tbar  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.QTbar = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.sigma_b = namelist['forcing']['sigma_b']
		self.k_a = namelist['forcing']['k_a']
		self.k_s = namelist['forcing']['k_s']
		self.k_f = namelist['forcing']['1.0']
		self.DT_y = namelist['forcing']['DT_y']
		self.Dtheta_z = namelist['forcing']['lapse_rate']
		self.cp = namelist['thermodynamics']['heat_capacity']
		self.Rd = namelist['thermodynamics']['ideal_gas_constant']
		self.kappa = self.Rd/self.cp
		self.k_T = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		for k in range(Gr.n_layers):
			self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z**2*
				np.log(np.divide(PV.P.values[:,:,k],Gr.p_ref))*np.cos(np.radians(Gr.lat))**2)*(np.divide(PV.P.values[:,:,k],Gr.p_ref))**self.kappa

		self.QTbar = np.zeros((Gr.nlats, Gr.nlons,Gr.n_layers))
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_zonal_mean('zonal_mean_QT_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_QT_eq')
		return

	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		# Initialise the random forcing function
		# F0 = np.zeros(Gr.Spharmt.nlm, dtype = np.complex)
		# fr = spf.sphForcing(Gr.nlats, Gr.nlons, Gr.truncation_number, Gr.rsphere,lmin= 98, lmax= 102, magnitude = 5, correlation = 0.)

		# Field initialisation
		for k in range(Gr.n_layers):
			p_ratio_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
			sigma_ratio_k = np.clip(np.divide(np.subtract(p_ratio_k,self.sigma_b),(1.0-self.sigma_b)) ,0.0, None)
			cos4_lat = np.power(np.cos(Gr.lat),4.0)
			self.k_T[:,:,k] = np.multiply(np.multiply((self.k_s-self.k_a),sigma_ratio_k),cos4_lat)
			self.k_T[:,:,k] = np.add(self.k_a,self.k_T[:,:,k])
			self.k_v[:,:,k] = np.multiply(self.k_f,sigma_ratio_k)

			self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z**2*
				np.log(np.divide(PV.P.values[:,:,k],Gr.p_ref))*np.cos(np.radians(Gr.lat))**2)*(np.divide(PV.P.values[:,:,k],Gr.p_ref))**self.kappa

			self.Tbar[:,:,k] = np.clip(self.Tbar[:,:,k],200.0,350.0)
			self.QTbar[:,:,k] = np.multiply(0.0,self.Tbar[:,:,k])

			u_forcing = np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k])
			v_forcing = np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])
			Vorticity_forcing, Divergece_forcing = Gr.Spharmt.getvrtdivspec(u_forcing, v_forcing)
			PV.Divergence.forcing[:,k] = - Divergece_forcing
			PV.Vorticity.forcing[:,k]  = - Vorticity_forcing
			PV.T.forcing[:,k]          = - Gr.Spharmt.grdtospec(np.multiply(self.k_T[:,:,k],(np.subtract(PV.T.values[:,:,k],self.Tbar[:,:,k]))))
		return


	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'T_eq', self.Tbar)
		Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'QT_eq',   self.QTbar)
		return

	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_zonal_mean('zonal_mean_QT_eq', self.QTbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_QT_eq', self.QTbar, TS.t)
		return

cdef class Forcing_HelzSuarez_moist:
	def __init__(self, Gr):
		self.Tbar = np.zeros(Gr.nx,Gr.ny, Gr.nz, dtype=np.double, order='c')
		self.QTbar = np.zeros(Gr.nx,Gr.ny, Gr.nz, dtype=np.double, order='c')
		return

	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		# constants from Held & Suarez (1994)
		self.PV = PrognosticVariables.PrognosticVariables(Gr)
		self.DV = DiagnosticVariables.DiagnosticVariables(Gr)
		self.Tbar  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_T   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.QTbar = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.sigma_b = namelist['forcing']['sigma_b']
		self.k_a = namelist['forcing']['k_a']
		self.k_s = namelist['forcing']['k_s']
		self.k_f = namelist['forcing']['1.0']
		self.DT_y = namelist['forcing']['DT_y']
		self.Dtheta_z = namelist['forcing']['lapse_rate']
		self.cp = namelist['thermodynamics']['heat_capacity']
		self.Rd = namelist['thermodynamics']['ideal_gas_constant']
		self.kappa = self.Rd/self.cp
		self.k_T = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.k_v = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		for k in range(Gr.n_layers):
			self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z**2*
				np.log(np.divide(PV.P.values[:,:,k],Gr.p_ref))*np.cos(np.radians(Gr.lat))**2)*(np.divide(PV.P.values[:,:,k],Gr.p_ref))**self.kappa

		self.QTbar = np.zeros((Gr.nlats, Gr.nlons,Gr.n_layers))
		return

	cpdef initialize_io(self, NetCDFIO_Stats Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_zonal_mean('zonal_mean_QT_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_QT_eq')
		return

	cpdef update(self, TimeStepping TS, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		# Initialise the random forcing function
		# F0 = np.zeros(Gr.Spharmt.nlm, dtype = np.complex)
		# fr = spf.sphForcing(Gr.nlats, Gr.nlons, Gr.truncation_number, Gr.rsphere,lmin= 98, lmax= 102, magnitude = 5, correlation = 0.)

		# Field initialisation
		for k in range(Gr.n_layers):
			p_ratio_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
			sigma_ratio_k = np.clip(np.divide(np.subtract(p_ratio_k,self.sigma_b),(1.0-self.sigma_b)) ,0.0, None)
			cos4_lat = np.power(np.cos(Gr.lat),4.0)
			self.k_T[:,:,k] = np.multiply(np.multiply((self.k_s-self.k_a),sigma_ratio_k),cos4_lat)
			self.k_T[:,:,k] = np.add(self.k_a,self.k_T[:,:,k])
			self.k_v[:,:,k] = np.multiply(self.k_f,sigma_ratio_k)

			self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z**2*
				np.log(np.divide(PV.P.values[:,:,k],Gr.p_ref))*np.cos(np.radians(Gr.lat))**2)*(np.divide(PV.P.values[:,:,k],Gr.p_ref))**self.kappa

			self.Tbar[:,:,k] = np.clip(self.Tbar[:,:,k],200.0,350.0)
			self.QTbar[:,:,k] = np.multiply(0.0,self.Tbar[:,:,k])

			u_forcing = np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k])
			v_forcing = np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])
			Vorticity_forcing, Divergece_forcing = Gr.Spharmt.getvrtdivspec(u_forcing, v_forcing)
			PV.Divergence.forcing[:,k] = - Divergece_forcing
			PV.Vorticity.forcing[:,k]  = - Vorticity_forcing
			PV.T.forcing[:,k]          = - Gr.Spharmt.grdtospec(np.multiply(self.k_T[:,:,k],(np.subtract(PV.T.values[:,:,k],self.Tbar[:,:,k]))))
		return


	cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'T_eq', self.Tbar)
		Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'QT_eq',   self.QTbar)
		return

	cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_zonal_mean('zonal_mean_QT_eq', self.QTbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_QT_eq', self.QTbar, TS.t)
		return