
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
		self.Tbar = np.zeros((Gr.nx,Gr.ny, Gr.nz), dtype=np.double, order='c')
		self.QTbar = np.zeros((Gr.nx,Gr.ny, Gr.nz), dtype=np.double, order='c')
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
		self.QTbar = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers), dtype=np.double, order='c')
		self.sigma_b = namelist['forcing']['sigma_b']
		self.k_a = namelist['forcing']['k_a']
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
		# for k in range(Gr.n_layers):
		# 	self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z*
		# 		np.log(PV.P.values[:,:,k]/Gr.p_ref)*np.cos(np.radians(Gr.lat))**2)*(PV.P.values[:,:,k]/Gr.p_ref)**self.kappa

		self.QTbar = np.zeros((Gr.nlats, Gr.nlons,Gr.n_layers))

		return

	def initialize_io(self, Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_zonal_mean('zonal_mean_QT_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_QT_eq')
		return

	def update(self, TS, Gr, PV, DV, namelist):
		# Initialise the random forcing function
		# F0 = np.zeros(Gr.SphericalGrid.nlm, dtype = np.complex)
		# fr = spf.sphForcing(Gr.nlats, Gr.nlons, Gr.truncation_number, Gr.rsphere,lmin= 98, lmax= 102, magnitude = 5, correlation = 0.)

		# k=0
		# sigma_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
		# sigma_ratio_k = np.clip(np.divide(sigma_k-self.sigma_b,(1.0-self.sigma_b)) ,0.1, None)
		# cos4_lat = np.power(np.cos(Gr.lat),4.0)
		# self.k_T[:,:,k] = np.multiply(np.multiply((self.k_s-self.k_a),sigma_ratio_k),cos4_lat)
		# self.k_T[:,:,k] = np.add(self.k_a,self.k_T[:,:,k])
		# self.k_v[:,:,k] = np.multiply(self.k_f,sigma_ratio_k)
		# print(k ,np.max(np.abs(self.k_T[:,:,k])))
		# print(k ,np.max(np.abs(self.k_v[:,:,k])))

		# self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z*
		# 	np.log(PV.P.values[:,:,k]/Gr.p_ref)*np.cos(np.radians(Gr.lat))**2)*(PV.P.values[:,:,k]/Gr.p_ref)**self.kappa

		# self.Tbar[:,:,k] = np.clip(self.Tbar[:,:,k],200.0,350.0)
		# self.QTbar[:,:,k] = np.multiply(0.0,self.Tbar[:,:,k])

		# u_forcing = np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k])
		# v_forcing = np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])
		# Vorticity_forcing, Divergece_forcing = Gr.SphericalGrid.getvrtdivspec(u_forcing, v_forcing)
		# PV.Divergence.forcing[:,k] = - np.multiply(Divergece_forcing, 0.0)
		# PV.Vorticity.forcing[:,k]  = - np.multiply(Vorticity_forcing, 0.0)
		# PV.T.forcing[:,k]          = - np.multiply(Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],(PV.T.values[:,:,k] - self.Tbar[:,:,k]))), 0.0)

		# k=1
		# sigma_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
		# sigma_ratio_k = np.clip(np.divide(sigma_k-self.sigma_b,(1.0-self.sigma_b)) ,0.0, None)
		# cos4_lat = np.power(np.cos(Gr.lat),4.0)
		# self.k_T[:,:,k] = np.multiply(np.multiply((self.k_s-self.k_a),sigma_ratio_k),cos4_lat)
		# self.k_T[:,:,k] = np.add(self.k_a,self.k_T[:,:,k])
		# self.k_v[:,:,k] = np.multiply(self.k_f,sigma_ratio_k)
		# print(k ,np.max(np.abs(self.k_T[:,:,k])))
		# print(k ,np.max(np.abs(self.k_v[:,:,k])))

		# self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z*
		# 	np.log(PV.P.values[:,:,k]/Gr.p_ref)*np.cos(np.radians(Gr.lat))**2)*(PV.P.values[:,:,k]/Gr.p_ref)**self.kappa

		# self.Tbar[:,:,k] = np.clip(self.Tbar[:,:,k],200.0,350.0)
		# self.QTbar[:,:,k] = np.multiply(0.0,self.Tbar[:,:,k])

		# u_forcing = np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k])
		# v_forcing = np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])
		# Vorticity_forcing, Divergece_forcing = Gr.SphericalGrid.getvrtdivspec(u_forcing, v_forcing)
		# PV.Divergence.forcing[:,k] = - np.multiply(Divergece_forcing, 0.0)
		# PV.Vorticity.forcing[:,k]  = - np.multiply(Vorticity_forcing, 0.0)
		# PV.T.forcing[:,k]          = - np.multiply(Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],(PV.T.values[:,:,k] - self.Tbar[:,:,k]))), 0.0)

		# k=2
		# sigma_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
		# sigma_ratio_k = np.clip(np.divide(sigma_k-self.sigma_b,(1.0-self.sigma_b)) ,0.0, None)
		# cos4_lat = np.power(np.cos(Gr.lat),4.0)
		# self.k_T[:,:,k] = np.multiply(np.multiply((self.k_s-self.k_a),sigma_ratio_k),cos4_lat)
		# self.k_T[:,:,k] = np.add(self.k_a,self.k_T[:,:,k])
		# self.k_v[:,:,k] = np.multiply(self.k_f,sigma_ratio_k)
		# print(k ,np.max(np.abs(self.k_T[:,:,k])))
		# print(k ,np.max(np.abs(self.k_v[:,:,k])))

		# self.Tbar[:,:,k]  = (315.0-self.DT_y*np.sin(np.radians(Gr.lat))**2-self.Dtheta_z*
		# 	np.log(PV.P.values[:,:,k]/Gr.p_ref)*np.cos(np.radians(Gr.lat))**2)*(PV.P.values[:,:,k]/Gr.p_ref)**self.kappa

		# self.Tbar[:,:,k] = np.clip(self.Tbar[:,:,k],200.0,350.0)
		# self.QTbar[:,:,k] = np.multiply(0.0,self.Tbar[:,:,k])

		# u_forcing = np.multiply(self.k_v[:,:,k],DV.U.values[:,:,k])
		# v_forcing = np.multiply(self.k_v[:,:,k],DV.V.values[:,:,k])
		# Vorticity_forcing, Divergece_forcing = Gr.SphericalGrid.getvrtdivspec(u_forcing, v_forcing)
		# PV.Divergence.forcing[:,k] = - np.multiply(Divergece_forcing, 0.0)
		# PV.Vorticity.forcing[:,k]  = - np.multiply(Vorticity_forcing, 0.0)
		# PV.T.forcing[:,k]          = - np.multiply(Gr.SphericalGrid.grdtospec(np.multiply(self.k_T[:,:,k],(PV.T.values[:,:,k] - self.Tbar[:,:,k]))), 0.0)

		# josef
		p0 = Gr.p_ref
		sigma_b=0.7      # sigma coordiantes as sigma=p/ps
		k_a = 1./40./(3600.*24)     # [1/s]
		k_b = 1./10./(3600.*24)     # [1/s]
		k_s = 1./4./(3600.*24)      # [1/s]
		k_f = 1./(3600.*24)         # [1/s]
		DT_y= 60.        # Characteristic temperature change in meridional direction [K]
		Dtheta_z = 10.   # Characteristic potential temperature change in vertical [K]
		Tbar1=np.zeros_like(PV.T.values[:,:,0])
		Tbar2=np.zeros_like(PV.T.values[:,:,1])
		Tbar3=np.zeros_like(PV.T.values[:,:,2])

		for jj in np.arange(0,Gr.nlons,1):
			Tbar1[:,jj]=(315.-self.DT_y*np.sin(Gr.lat[:,0])**2-self.Dtheta_z*np.log((PV.P.values[:,jj,0]+PV.P.values[:,jj,1])/(2.*p0))*np.cos(Gr.lat[:,0])**2)*((PV.P.values[:,jj,0]+PV.P.values[:,jj,1])/(2.*p0))**Gr.kappa
			Tbar2[:,jj]=(315.-self.DT_y*np.sin(Gr.lat[:,0])**2-self.Dtheta_z*np.log((PV.P.values[:,jj,1]+PV.P.values[:,jj,2])/(2.*p0))*np.cos(Gr.lat[:,0])**2)*((PV.P.values[:,jj,1]+PV.P.values[:,jj,2])/(2.*p0))**Gr.kappa
			Tbar3[:,jj]=(315.-self.DT_y*np.sin(Gr.lat[:,0])**2-self.Dtheta_z*np.log((PV.P.values[:,jj,2]+PV.P.values[:,jj,3])/(2.*p0))*np.cos(Gr.lat[:,0])**2)*((PV.P.values[:,jj,2]+PV.P.values[:,jj,3])/(2.*p0))**Gr.kappa

		Tbar1[Tbar1<=200.]=200. # minimum equilibrium Temperature is 200 K
		Tbar2[Tbar2<=200.]=200. # minimum equilibrium Temperature is 200 K
		Tbar3[Tbar3<=200.]=200. # minimum equilibrium Temperature is 200 K

		# from Held&Suarez (1994)
		sigma1=np.divide((PV.P.values[:,:,0]+PV.P.values[:,:,1])/2.,PV.P.values[:,:,Gr.n_layers])
		sigma2=np.divide((PV.P.values[:,:,1]+PV.P.values[:,:,2])/2.,PV.P.values[:,:,Gr.n_layers])
		sigma3=np.divide((PV.P.values[:,:,2]+PV.P.values[:,:,3])/2.,PV.P.values[:,:,Gr.n_layers])

		sigma_ratio1=np.clip(np.divide(sigma1-sigma_b,1-sigma_b),0,None)
		sigma_ratio2=np.clip(np.divide(sigma2-sigma_b,1-sigma_b),0,None)
		sigma_ratio3=np.clip(np.divide(sigma3-sigma_b,1-sigma_b),0,None)
		k_T1=k_a+(k_s-k_a)*np.multiply(sigma_ratio1,np.power(np.cos(Gr.lat),4))
		k_T2=k_a+(k_s-k_a)*np.multiply(sigma_ratio2,np.power(np.cos(Gr.lat),4))
		k_T3=k_a+(k_s-k_a)*np.multiply(sigma_ratio3,np.power(np.cos(Gr.lat),4))
		k_v1=k_b+k_f*sigma_ratio1
		k_v2=k_b+k_f*sigma_ratio2
		k_v3=k_b+k_f*sigma_ratio3

		#wind forcing from H&S
		fu1=-np.multiply(k_v1,DV.U.values[:,:,0])
		fu2=-np.multiply(k_v2,DV.U.values[:,:,1])
		fu3=-np.multiply(k_v3,DV.U.values[:,:,2])
		fv1=-np.multiply(k_v1,DV.V.values[:,:,0])
		fv2=-np.multiply(k_v2,DV.V.values[:,:,1])
		fv3=-np.multiply(k_v3,DV.V.values[:,:,2])
		fvrtsp1, fdivsp1 = Gr.SphericalGrid.getvrtdivspec(fu1, fv1)
		fvrtsp2, fdivsp2 = Gr.SphericalGrid.getvrtdivspec(fu2, fv2)
		fvrtsp3, fdivsp3 = Gr.SphericalGrid.getvrtdivspec(fu3, fv3)

		k=0
		PV.Divergence.forcing[:,k] =  fdivsp1
		PV.Vorticity.forcing[:,k]  =  fvrtsp1
		PV.T.forcing[:,k]          = -Gr.SphericalGrid.grdtospec(np.multiply(k_T1,(PV.T.values[:,:,0]-Tbar1)))
		k=1
		PV.Divergence.forcing[:,k] =  fdivsp2
		PV.Vorticity.forcing[:,k]  =  fvrtsp2
		PV.T.forcing[:,k]          = -Gr.SphericalGrid.grdtospec(np.multiply(k_T2,(PV.T.values[:,:,1]-Tbar2)))
		k=2
		PV.Divergence.forcing[:,k] = fdivsp3
		PV.Vorticity.forcing[:,k]  = fvrtsp3
		PV.T.forcing[:,k]          = -Gr.SphericalGrid.grdtospec(np.multiply(k_T3,(PV.T.values[:,:,2]-Tbar3)))

		return

	def io(self, Gr, TS, Stats):
		Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'T_eq', self.Tbar)
		Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'QT_eq',   self.QTbar)
		return

	def stats_io(self, TS, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar, TS.t)
		Stats.write_zonal_mean('zonal_mean_QT_eq', self.QTbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar, TS.t)
		Stats.write_meridional_mean('meridional_mean_QT_eq', self.QTbar, TS.t)
		return

class Forcing_HelzSuarez_moist:
	def __init__(self, Gr):
		self.Tbar = np.zeros(Gr.nx,Gr.ny, Gr.nz, dtype=np.double, order='c')
		self.QTbar = np.zeros(Gr.nx,Gr.ny, Gr.nz, dtype=np.double, order='c')
		return

	def initialize(self, Gr, PV, DV, namelist):
		# for k in range(self.n_layers):
		# 	self.Tbar[:,:,k]  = (315.0-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p1/ps[:,jj])*np.cos(y)**2)*(p1/ps[:,jj])**kappa
	 #        self.QTbar[:,:,k]  = 0.0
		return

	def initialize_io(self, Stats):
		Stats.add_variable('relaxation_temp')
		Stats.add_variable('relaxation_qt')
		return

	def update():
		# # Initialise the random forcing function
		# F0 = np.zeros(x.nlm, dtype = np.complex)
		# fr = spf.sphForcing(nlons,nlats,truncation_number,rsphere,lmin= 98, lmax= 102, magnitude = 5, correlation = 0.)
		# #fr = spf.sphForcing(nlons,nlats,truncation_number,rsphere,lmin= 68, lmax=  72, magnitude = 5, correlation = 0.)
		# #fr = spf.sphForcing(nlons,nlats,truncation_number,rsphere,lmin= 48, lmax=  52, magnitude = 5, correlation = 0.)
		# #fr = spf.sphForcing(nlons,nlats,truncation_number,rsphere,lmin= 18, lmax=  22, magnitude = 5, correlation = 0.)
		# # lmin and lmax are the wavenumbers range which is forced
		# # Field initialisation
		# for k in range(self.n_layers):
		# 	self.Tbar[:,:,k]  = Tbar1[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p1/ps[:,jj])*np.cos(y)**2)*(p1/ps[:,jj])**kappa
	 #        self.QTbar[:,:,k]  = 0.0
		return

	def io(self, Stats):
		Stats.write_variable('relaxation_temp', self.Tbar)
		Stats.write_variable('relaxation_qt',   self.QTbar)
		return
