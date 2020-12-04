import numpy as np

class TimeStepping:
	def __init__(self, Pr, namelist):
		self.dt = namelist['timestepping']['dt']
		self.t_max = namelist['timestepping']['t_max']
		return

	def initialize(self):
		self.t = 0.0
		self.ncycle = 0
		return

	def update(self, Pr, Gr, PV, DV, DF, namelist):
		# self.CFL_limiter(Pr, Gr, DV, namelist)
		dt = 100.0
		F_Divergence = PV.Divergence.spectral
		F_Vorticity  = PV.Vorticity.spectral
		F_T          = PV.T.spectral
		F_P          = PV.P.spectral

        #forward euler, then 2nd and than 3rd order adams-bashforth time stepping
		if self.ncycle==0:
			PV.set_now_with_tendencies()
		    # Euler
		    # new                       old               tendency
			PV.Divergence.spectral = F_Divergence + dt*PV.Divergence.tendency
			PV.Vorticity.spectral  = F_Vorticity  + dt*PV.Vorticity.tendency
			PV.T.spectral          = F_T          + dt*PV.T.tendency
			PV.P.spectral          = F_P          + dt*PV.P.tendency
		elif self.ncycle==1:
        	#2nd order AB
        	#           new                old               tendency                            tendency now
			PV.Divergence.spectral = F_Divergence + dt*((3.0/2.0)*PV.Divergence.tendency - (1.0/2.0)*PV.Divergence.now)
			PV.Vorticity.spectral  = F_Vorticity  + dt*((3.0/2.0)*PV.Vorticity.tendency  - (1.0/2.0)*PV.Vorticity.now)
			PV.T.spectral          = F_T          + dt*((3.0/2.0)*PV.T.tendency          - (1.0/2.0)*PV.T.now)
			PV.P.spectral          = F_P          + dt*((3.0/2.0)*PV.P.tendency          - (1.0/2.0)*PV.P.now)
		else:
        	#3nd order AB
        	#           new                old               tendency                           tendency now                         tendency old
			PV.Divergence.spectral = F_Divergence + dt*((23.0/12.0)*PV.Divergence.tendency - (16.0/12.0)*PV.Divergence.now + (5.0/12.0)*PV.Divergence.old)
			PV.Vorticity.spectral  = F_Vorticity  + dt*((23.0/12.0)*PV.Vorticity.tendency  - (16.0/12.0)*PV.Vorticity.now  + (5.0/12.0)*PV.Vorticity.old)
			PV.T.spectral          = F_T          + dt*((23.0/12.0)*PV.T.tendency          - (16.0/12.0)*PV.T.now          + (5.0/12.0)*PV.T.old)
			PV.P.spectral          = F_P          + dt*((23.0/12.0)*PV.P.tendency          - (16.0/12.0)*PV.P.now          + (5.0/12.0)*PV.P.old)

		DF.update(Pr, Gr, PV, self.dt, namelist)
		self.t = self.t+self.dt
		PV.set_old_with_now()
		PV.set_now_with_tendencies()

		self.ncycle += 1
		return

	def CFL_limiter(self, Pr, Gr, DV, namelist):
		# consider calling this every some time to save computation
		CFL_limit = namelist['timestepping']['CFL_limit']
		dt = namelist['timestepping']['dt']
		dx = 2.0*np.divide(np.pi,Pr.nlats)*Pr.rsphere
		dy = 2.0*np.divide(np.pi,Pr.nlons)*Pr.rsphere
		dp = np.max([Pr.p_ref-Pr.p3,Pr.p3-Pr.p2,Pr.p2-Pr.p1])
		zonal_timescale = np.min(dx)/np.max(np.abs(DV.U.values) + 1e-10)
		meridional_timescale = np.min(dy)/np.max(np.abs(DV.V.values) + 1e-10)
		vertical_timescale = np.min(dp)/np.max(np.abs(DV.Wp.values) + 1e-10)
		self.dt = np.minimum(dt, CFL_limit*np.max([zonal_timescale ,meridional_timescale]))

		return



