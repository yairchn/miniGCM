# from Grid import Grid
import numpy as np
# from NetCDFIO import NetCDFIO_Stats
# from TimeStepping import TimeStepping
import sys
# from Parameters import Parameters
# import sphericalForcing as spf


def ForcingFactory(namelist):
	if namelist['forcing']['forcing_model'] == 'None':
		return ForcingNone(namelist)
	elif namelist['forcing']['forcing_model'] == 'HeldSuarez':
		return HelzSuarez(namelist)
	else:
		print('case not recognized')
	return

class ForcingBase:
	def __init__(self, namelist):
		return
	def initialize(self, Pr, Gr, namelist):
		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.sin_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		self.cos_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		self.Tbar_layer = np.zeros((Pr.nlats, Pr.nlons), dtype=np.double, order='c')
		self.Tbar_meridional = np.zeros((Pr.nlons,Pr.n_layers), dtype=np.double, order='c')
		self.pressure_ratio_meridional = np.zeros(Pr.nlons, dtype=np.double, order='c')
		return
	def initialize_io(self, Stats):
		return
	def update(self, Pr, Gr, PV, DV):
		return
	def io(self, Pr, TS, Stats):
		return
	def stats_io(self, Stats):
		return

class ForcingNone(ForcingBase):
	def __init__(self, namelist):
		ForcingBase.__init__(self, namelist)
		return
	def initialize(self, Pr, Gr, namelist):
		return
	def initialize_io(self, Stats):
		return
	def update(self, Pr, Gr, PV, DV):
		return
	def io(self, Pr, TS, Stats):
		return
	def stats_io(self, Stats):
		return

class HelzSuarez(ForcingBase):
	def __init__(self, namelist):
		ForcingBase.__init__(self, namelist)
		return

	def initialize(self, Pr, Gr, namelist):
		nx = Pr.nlats
		ny = Pr.nlons

		self.Tbar = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.float64, order='c')
		self.sin_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		self.cos_lat = np.zeros((Pr.nlats, Pr.nlons), dtype=np.float64, order='c')
		self.noise = namelist['forcing']['noise']
		Pr.Fo_noise_amplitude  = namelist['forcing']['noise_amplitude']
		Pr.Fo_noise_magnitude  = namelist['forcing']['noise_magnitude']
		Pr.Fo_noise_correlation = namelist['forcing']['noise_correlation']
		Pr.Fo_noise_type  = namelist['forcing']['noise_type']
		Pr.Fo_noise_lmin  = namelist['forcing']['min_noise_wavenumber']
		Pr.Fo_noise_lmax  = namelist['forcing']['max_noise_wavenumber']
		self.k_T   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_Q   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.k_v   = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		for i in range(nx):
			for j in range(ny):
				self.sin_lat[i,j] = np.sin(Gr.lat[i,j])
				self.cos_lat[i,j] = np.cos(Gr.lat[i,j])
		return

	def initialize_io(self, Stats):
		Stats.add_zonal_mean('zonal_mean_T_eq')
		Stats.add_meridional_mean('meridional_mean_T_eq')
		return

	def update(self, Pr, Gr, PV, DV):
		nx = Pr.nlats
		ny = Pr.nlons
		nl = Pr.n_layers
		nlm = Gr.SphericalGrid.nlm
		forcing_noise = np.zeros((nx,ny)   ,dtype=np.float64, order='c')
		sp_noise = np.zeros(nlm    ,dtype=np.complex, order='c')

		for k in range(Pr.n_layers):
			self.pressure_ratio_meridional=(np.mean(PV.P.values[:,:,k], axis=1)+np.mean(PV.P.values[:,:,k+1],axis=1))/(2.0*Pr.p_ref)
			self.Tbar_meridional=((315.0-Pr.DT_y*np.sin(Gr.lat[:,0])**2-Pr.Dtheta_z*np.log(self.pressure_ratio_meridional)*np.cos(Gr.lat[:,0])**2)*(self.pressure_ratio_meridional)**Pr.kappa)
			self.Tbar[:,:,k] = np.repeat(self.Tbar_meridional[:, np.newaxis], Pr.nlons, axis=1)
			self.Tbar[:,:,k][self.Tbar[:,:,k]<=200.0]=200.0
			sigma=np.divide((PV.P.values[:,:,k]+PV.P.values[:,:,k+1])/2.,PV.P.values[:,:,Pr.n_layers])
			sigma_ratio=np.clip(np.divide(sigma-Pr.sigma_b,1-Pr.sigma_b),0,None)
			self.k_T[:,:,k] = Pr.k_a+(Pr.k_s-Pr.k_a)*np.multiply(sigma_ratio,np.power(np.cos(Gr.lat),4))
			self.k_v[:,:,k] = Pr.k_a+Pr.k_f*sigma_ratio
			DV.U.forcing[:,:,k] = - self.k_v[:,:,k] *  DV.U.values[:,:,k]
			DV.V.forcing[:,:,k] = - self.k_v[:,:,k] *  DV.V.values[:,:,k]
			PV.T.forcing[:,:,k] = -self.k_T[:,:,k] * (PV.T.values[:,:,k] - self.Tbar[:,:,k])

		if self.noise:
			#print('Add stochastic forcing for vorticity equation')
			F0=np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
			fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,
				                Pr.Fo_noise_lmin, Pr.Fo_noise_lmax, Pr.Fo_noise_magnitude,
				                correlation =Pr.Fo_noise_correlation, noise_type=Pr.Fo_noise_type)

			forcing_noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(F0))*Pr.Fo_noise_amplitude
			sp_noise = Gr.SphericalGrid.grdtospec(forcing_noise)
			PV.Vorticity.sp_forcing[:,Pr.n_layers-1] = sp_noise # forcing only in lowest layer
		return



	def io(self, Pr, TS, Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	def stats_io(self, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar)
		return