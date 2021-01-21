import numpy as np
import sys
import cython
from Grid cimport Grid
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Diffusion cimport Diffusion
from Parameters cimport Parameters

cdef class TimeStepping:
	def __init__(self, namelist):
		self.dt = namelist['timestepping']['dt']
		self.t_max = namelist['timestepping']['t_max']*24.0*3600.0 # sec/day
		return

	cpdef initialize(self):
		self.t = 0.0
		self.ncycle = 0
		return

	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist):
		self.dt = namelist['timestepping']['dt']
		self.CFL_limiter(Pr, Gr, DV, namelist)
		# Override
		self.dt = namelist['timestepping']['dt']

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
			PV.Divergence.spectral = np.add(F_Divergence, np.multiply(self.dt,PV.Divergence.tendency))
			PV.Vorticity.spectral  = np.add(F_Vorticity , np.multiply(self.dt,PV.Vorticity.tendency))
			PV.T.spectral          = np.add(F_T         , np.multiply(self.dt,PV.T.tendency))
			PV.QT.spectral         = np.add(F_QT        , np.multiply(self.dt,PV.QT.tendency))
			PV.P.spectral          = np.add(F_P         , np.multiply(self.dt,PV.P.tendency))
		elif self.ncycle==1:
			#2nd order AB
			#           new                old               tendency                            tendency now
			PV.Divergence.spectral = np.add(F_Divergence,np.multiply(self.dt, np.subtract(np.multiply(1.5,PV.Divergence.tendency), np.multiply(0.5,PV.Divergence.now))))
			PV.Vorticity.spectral  = np.add(F_Vorticity ,np.multiply(self.dt, np.subtract(np.multiply(1.5,PV.Vorticity.tendency ), np.multiply(0.5,PV.Vorticity.now))))
			PV.T.spectral          = np.add(F_T         ,np.multiply(self.dt, np.subtract(np.multiply(1.5,PV.T.tendency         ), np.multiply(0.5,PV.T.now))))
			PV.QT.spectral         = np.add(F_QT        ,np.multiply(self.dt, np.subtract(np.multiply(1.5,PV.QT.tendency        ), np.multiply(0.5,PV.QT.now))))
			PV.P.spectral          = np.add(F_P         ,np.multiply(self.dt, np.subtract(np.multiply(1.5,PV.P.tendency         ), np.multiply(0.5,PV.P.now))))
		else:
			#3nd order AB
			#           new                 old                                                                  tendency                                     tendency now                               tendency old
			PV.Divergence.spectral = np.add(F_Divergence,np.multiply(self.dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.Divergence.tendency),np.multiply(16.0/12.0,PV.Divergence.now)), np.multiply(5.0/12.0,PV.Divergence.old))))
			PV.Vorticity.spectral  = np.add(F_Vorticity ,np.multiply(self.dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.Vorticity.tendency) ,np.multiply(16.0/12.0,PV.Vorticity.now)) , np.multiply(5.0/12.0,PV.Vorticity.old))))
			PV.T.spectral          = np.add(F_T         ,np.multiply(self.dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.T.tendency)         ,np.multiply(16.0/12.0,PV.T.now))         , np.multiply(5.0/12.0,PV.T.old))))
			PV.QT.spectral         = np.add(F_QT        ,np.multiply(self.dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.QT.tendency)        ,np.multiply(16.0/12.0,PV.QT.now))        , np.multiply(5.0/12.0,PV.QT.old))))
			PV.P.spectral          = np.add(F_P         ,np.multiply(self.dt,np.add(np.subtract(np.multiply(23.0/12.0,PV.P.tendency)         ,np.multiply(16.0/12.0,PV.P.now))         , np.multiply(5.0/12.0,PV.P.old))))

		DF.update(Pr, Gr, PV, self.dt)
		self.t = self.t+self.dt
		PV.set_old_with_now()
		PV.set_now_with_tendencies()

		self.ncycle += 1
		return

	cpdef CFL_limiter(self, Parameters Pr, Grid Gr, DiagnosticVariables DV, namelist):
		cdef:
			double zonal_timescale
			double meridional_timescale

		CFL_limit = namelist['timestepping']['CFL_limit']
		zonal_timescale      = np.max(Gr.dx/(np.max(np.abs(DV.U.values) + 1.0)))
		meridional_timescale = np.max(Gr.dy/(np.max(np.abs(DV.V.values) + 1.0)))
		self.dt = np.minimum(self.dt, CFL_limit*np.min([zonal_timescale ,meridional_timescale]))
		return
