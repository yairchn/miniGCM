import numpy as np
import sys
import cython
from Grid cimport Grid
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Diffusion cimport Diffusion
import Parameters

cdef class TimeStepping:
	def __init__(self, namelist):
		self.dt = namelist['timestepping']['dt']
		self.t_max = namelist['timestepping']['t_max']
		return

	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist):
		self.t = 0.0
		self.ncycle = 0
		return

	cpdef update(self, Grid Gr,  PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist):
		# self.dt = CFL_limiter(self, Gr, PV, DV, DF, namelist)
		CFL_limit = namelist['timestepping']['CFL_limit']
		dt = namelist['timestepping']['dt']
		self.dx = 2.0*np.pi/Gr.nlats*Gr.rsphere
		self.dy = 2.0*np.pi/Gr.nlons*Gr.rsphere
		self.dp = np.max([Gr.p_ref-Gr.p3,Gr.p3-Gr.p2,Gr.p2-Gr.p1])
		zonal_timescale = np.min(self.dx)/np.max(np.abs(DV.U.values) + 1e-10)
		meridional_timescale = np.min(self.dy)/np.max(np.abs(DV.V.values) + 1e-10)
		vertical_timescale = np.min(self.dp)/np.max(np.abs(DV.Wp.values) + 1e-10)
		# dt = np.minimum(dt, CFL_limit*np.max([zonal_timescale ,meridional_timescale ,vertical_timescale]))
		dt=100.0 # overwrite for now

		F_Divergence = PV.Divergence.spectral
		F_Vorticity  = PV.Vorticity.spectral
		F_T          = PV.T.spectral
		F_QT         = PV.QT.spectral
		F_P          = PV.P.spectral

        #forward euler, then 2nd and than 3rd order adams-bashforth time stepping
		if self.ncycle==0:
			PV.set_now_with_tendencies()
		    # Euler
		    # new                       old               tendency
			PV.Divergence.spectral = np.add(F_Divergence, np.multiply(dt,PV.Divergence.tendency))
			PV.Vorticity.spectral  = np.add(F_Vorticity , np.multiply(dt,PV.Vorticity.tendency))
			PV.T.spectral          = np.add(F_T         , np.multiply(dt,PV.T.tendency))
			PV.QT.spectral         = np.add(F_QT        , np.multiply(dt,PV.QT.tendency))
			PV.P.spectral          = np.add(F_P         , np.multiply(dt,PV.P.tendency))
		elif self.ncycle==1:
        	#2nd order AB
        	#           new                old               tendency                            tendency now
			PV.Divergence.spectral = np.add(F_Divergence,np.multiply(dt, np.subtract(np.multiply(1.5,PV.Divergence.tendency), np.multiply(0.5,PV.Divergence.now))))
			PV.Vorticity.spectral  = np.add(F_Vorticity ,np.multiply(dt, np.subtract(np.multiply(1.5,PV.Vorticity.tendency ), np.multiply(0.5,PV.Vorticity.now))))
			PV.T.spectral          = np.add(F_T         ,np.multiply(dt, np.subtract(np.multiply(1.5,PV.T.tendency         ), np.multiply(0.5,PV.T.now))))
			PV.QT.spectral         = np.add(F_QT        ,np.multiply(dt, np.subtract(np.multiply(1.5,PV.QT.tendency        ), np.multiply(0.5,PV.QT.now))))
			PV.P.spectral          = np.add(F_P         ,np.multiply(dt, np.subtract(np.multiply(1.5,PV.P.tendency         ), np.multiply(0.5,PV.P.now))))
		else:
        	#3nd order AB
        	#           new                 old                                                                  tendency                                     tendency now                               tendency old
			PV.Divergence.spectral = np.add(F_Divergence,np.multiply(dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.Divergence.tendency),np.multiply(16.0/12.0,PV.Divergence.now)), np.multiply(5.0/12.0,PV.Divergence.old))))
			PV.Vorticity.spectral  = np.add(F_Vorticity ,np.multiply(dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.Vorticity.tendency) ,np.multiply(16.0/12.0,PV.Vorticity.now)) , np.multiply(5.0/12.0,PV.Vorticity.old))))
			PV.T.spectral          = np.add(F_T         ,np.multiply(dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.T.tendency)         ,np.multiply(16.0/12.0,PV.T.now))         , np.multiply(5.0/12.0,PV.T.old))))
			PV.QT.spectral         = np.add(F_QT        ,np.multiply(dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.QT.tendency)        ,np.multiply(16.0/12.0,PV.QT.now))        , np.multiply(5.0/12.0,PV.QT.old))))
			PV.P.spectral          = np.add(F_P         ,np.multiply(dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.P.tendency)         ,np.multiply(16.0/12.0,PV.P.now))         , np.multiply(5.0/12.0,PV.P.old))))

		DF.update(Pr, Gr, PV, self.dt, namelist)
		self.t = self.t+self.dt
		PV.set_old_with_now()
		PV.set_now_with_tendencies()

		self.ncycle += 1
		return

	cpdef CFL_limiter(self, Parameters Pr, Grid Gr, DiagnosticVariables DV, namelist):
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



