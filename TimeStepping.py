import numpy as np
from math import *
from PrognosticVariables import PrognosticVariables
import pylab as plt

class TimeStepping:
	def __init__(self, namelist):
		self.dt = namelist['timestepping']['dt']
		self.t_max = namelist['timestepping']['t_max']
		return

	def initialize(self, Gr, PV, DV, DF, namelist):
		self.t = 0.0
		self.ncycle = 0
		return

	def update(self, Gr, PV, DV, DF, namelist):
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
			PV.Divergence.spectral = F_Divergence + dt*PV.Divergence.tendency
			PV.Vorticity.spectral  = F_Vorticity  + dt*PV.Vorticity.tendency
			PV.T.spectral          = F_T          + dt*PV.T.tendency
			PV.QT.spectral         = F_QT         + dt*PV.QT.tendency
			PV.P.spectral          = F_P          + dt*PV.P.tendency
		elif self.ncycle==1:
        	#2nd order AB
        	#           new                old               tendency                            tendency now
			PV.Divergence.spectral = F_Divergence + dt*((3.0/2.0)*PV.Divergence.tendency - (1.0/2.0)*PV.Divergence.now)
			PV.Vorticity.spectral  = F_Vorticity  + dt*((3.0/2.0)*PV.Vorticity.tendency  - (1.0/2.0)*PV.Vorticity.now)
			PV.T.spectral          = F_T          + dt*((3.0/2.0)*PV.T.tendency          - (1.0/2.0)*PV.T.now)
			PV.QT.spectral         = F_QT         + dt*((3.0/2.0)*PV.QT.tendency         - (1.0/2.0)*PV.QT.now)
			PV.P.spectral          = F_P          + dt*((3.0/2.0)*PV.P.tendency          - (1.0/2.0)*PV.P.now)
		else:
        	#3nd order AB
        	#           new                old               tendency                           tendency now                         tendency old
			PV.Divergence.spectral = F_Divergence + dt*((23.0/12.0)*PV.Divergence.tendency - (16.0/12.0)*PV.Divergence.now + (5.0/12.0)*PV.Divergence.old)
			PV.Vorticity.spectral  = F_Vorticity  + dt*((23.0/12.0)*PV.Vorticity.tendency  - (16.0/12.0)*PV.Vorticity.now  + (5.0/12.0)*PV.Vorticity.old)
			PV.T.spectral          = F_T          + dt*((23.0/12.0)*PV.T.tendency          - (16.0/12.0)*PV.T.now          + (5.0/12.0)*PV.T.old)
			PV.QT.spectral         = F_QT         + dt*((23.0/12.0)*PV.QT.tendency         - (16.0/12.0)*PV.QT.now         + (5.0/12.0)*PV.QT.old)
			PV.P.spectral          = F_P          + dt*((23.0/12.0)*PV.P.tendency          - (16.0/12.0)*PV.P.now          + (5.0/12.0)*PV.P.old)

		DF.update(Gr, PV, namelist, self.dt)
		self.t = self.t+self.dt
		PV.set_old_with_now()
		PV.set_now_with_tendencies()

		self.ncycle += 1
		return

	# def CFL_limiter(self, Gr, PV, DV, DF, namelist):
	# 	# YAIR
	# 	# (1) - find out how to limit a CFL in a spectral model
	# 	# (2) - define the pointwise U/DX ratio and find its maximum value for i,j,k
	# 	# in Grid line 40 self.longitute , self.latitute  are defines. Take diff or gradient of these to get the differneces matrix.
	# 	return dt



