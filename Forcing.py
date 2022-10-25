from Grid cimport Grid
import numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from TimeStepping cimport TimeStepping
import sys
from Parameters cimport Parameters
import sphericalForcing as spf
from libc.math cimport pow, log, sin, cos, fmax
import sphericalForcing as spf


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
		def:
			Py_ssize_t i,j
			Py_ssize_t nx = Pr.nlats
			Py_ssize_t ny = Pr.nlons

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
		for i in range(nx):
			for j in range(ny):
				self.sin_lat[i,j] = sin(Gr.lat[i,j])
				self.cos_lat[i,j] = cos(Gr.lat[i,j])
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

		for i in range(nx):
			for j in range(ny):
				for k in range(nl):
					p_half = (p[i,j,k]+p[i,j,k+1])/2.0
					self.Tbar[i,j,k] = fmax(((Pr.T_equator - Pr.DT_y*self.sin_lat[i,j]*self.sin_lat[i,j] -
						Pr.Dtheta_z*log(p_half/Pr.p_ref)*self.cos_lat[i,j]*self.cos_lat[i,j])*
						pow(p_half/Pr.p_ref, Pr.kappa)),200.0)


					sigma_ratio = fmax((p_half/p[i,j,nl]-Pr.sigma_b)/(1-Pr.sigma_b),0.0)
					k_T = Pr.k_a + (Pr.k_s-Pr.k_a)*sigma_ratio*pow(self.cos_lat[i,j],4.0)
					k_v = Pr.k_b +  Pr.k_f*sigma_ratio

					DV.U.forcing[i,j,k] = -k_v *  DV.U.forcing[i,j,k]
					DV.V.forcing[i,j,k] = -k_v *  DV.V.forcing[i,j,k]
					PV.T.forcing[i,j,k] = -k_T * (PV.T.values[i,j,k] - self.Tbar[i,j,k])

		if self.noise:
			#print('Add stochastic forcing for vorticity equation')
			F0=np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
			fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,
				                Pr.Fo_noise_lmin, Pr.Fo_noise_lmax, Pr.Fo_noise_magnitude,
				                correlation =Pr.Fo_noise_correlation, noise_type=Pr.Fo_noise_type)

			forcing_noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(F0))*Pr.Fo_noise_amplitude
			sp_noise = Gr.SphericalGrid.grdtospec(forcing_noise.base)
			PV.Vorticity.sp_forcing[:,Pr.n_layers-1] = sp_noise # forcing only in lowest layer
		return



	def io(self, Pr, TS, Stats):
		Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'T_eq', self.Tbar)
		return

	def stats_io(self, Stats):
		Stats.write_zonal_mean('zonal_mean_T_eq', self.Tbar)
		Stats.write_meridional_mean('meridional_mean_T_eq', self.Tbar)
		return