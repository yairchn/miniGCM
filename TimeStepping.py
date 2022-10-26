
import numpy as np
from Grid import Grid
# from PrognosticVariables import PrognosticVariables
from DiagnosticVariables import DiagnosticVariables
from Diffusion import Diffusion
from Parameters import Parameters

class TimeStepping:
	def __init__(self, namelist):
		self.dt = namelist['timestepping']['dt']
		self.t_max = namelist['timestepping']['t_max']*24.0*3600.0 # sec/day
		self.nt = self.t_max/self.dt
		return

	def initialize(self, Pr):
		if not Pr.restart:
			self.t = 0.0
		self.ncycle = 0
		return

	def update(self, Pr, Gr, PV, DV, DF, namelist):
		nl = Pr.n_layers
		nlm = Gr.SphericalGrid.nlm
		F_Divergence = np.zeros((nlm, nl), dtype = np.complex, order='c')
		F_Vorticity  = np.zeros((nlm, nl), dtype = np.complex, order='c')
		F_T          = np.zeros((nlm, nl), dtype = np.complex, order='c')
		F_QT         = np.zeros((nlm, nl), dtype = np.complex, order='c')
		F_P          = np.zeros((nlm),     dtype = np.complex, order='c')

		# self.CFL_limiter(Pr, Gr, DV, namelist)
		dt = namelist['timestepping']['dt']

		F_P          = PV.P.spectral[:,nl]
		F_Divergence = PV.Divergence.spectral
		F_Vorticity  = PV.Vorticity.spectral
		F_T          = PV.T.spectral
		if Pr.moist_index > 0.0:
			F_QT         = PV.QT.spectral

		if self.ncycle==0:
			PV.set_now_with_tendencies()
			for i in range(nlm):
				PV.P.spectral[i,nl]             = F_P[i]            + self.dt*PV.P.tendency[i,nl]
				for k in range(nl):
					# Euler
					# new                                old                      tendency
					PV.Divergence.spectral[i,k] = F_Divergence[i,k] + self.dt*PV.Divergence.tendency[i,k]
					PV.Vorticity.spectral[i,k]  = F_Vorticity[i,k]  + self.dt*PV.Vorticity.tendency[i,k]
					PV.T.spectral[i,k]          = F_T[i,k]          + self.dt*PV.T.tendency[i,k]
					if Pr.moist_index > 0.0:
						PV.QT.spectral[i,k]         = F_QT[i,k]         + self.dt*PV.QT.tendency[i,k]
		elif self.ncycle==1:
			for i in range(nlm):
				PV.P.spectral[i,nl]             = F_P[i]            + self.dt*(1.5*PV.P.tendency[i,nl]         - 0.5*PV.P.now[i,nl])
				for k in range(nl):
					#2nd order AB
					#           new                     old                        tendency                            tendency now
					PV.Divergence.spectral[i,k] = F_Divergence[i,k] + self.dt*(1.5*PV.Divergence.tendency[i,k] - 0.5*PV.Divergence.now[i,k])
					PV.Vorticity.spectral[i,k]  = F_Vorticity[i,k]  + self.dt*(1.5*PV.Vorticity.tendency[i,k]  - 0.5*PV.Vorticity.now[i,k])
					PV.T.spectral[i,k]          = F_T[i,k]          + self.dt*(1.5*PV.T.tendency[i,k]          - 0.5*PV.T.now[i,k])
					if Pr.moist_index > 0.0:
						PV.QT.spectral[i,k]         = F_QT[i,k]         + self.dt*(1.5*PV.QT.tendency[i,k]         - 0.5*PV.QT.now[i,k])
		else:
			for i in range(nlm):
				PV.P.spectral[i,nl]             = F_P[i]            + self.dt*(23.0/12.0*PV.P.tendency[i,nl]         - 16.0/12.0*PV.P.now[i,nl]         + 5.0/12.0*PV.P.old[i,nl]        )
				for k in range(nl):
					#3nd order AB
					#           new                    old                               tendency                                tendency now                         tendency old
					PV.Divergence.spectral[i,k] = F_Divergence[i,k] + self.dt*(23.0/12.0*PV.Divergence.tendency[i,k] - 16.0/12.0*PV.Divergence.now[i,k] + 5.0/12.0*PV.Divergence.old[i,k])
					PV.Vorticity.spectral[i,k]  = F_Vorticity[i,k]  + self.dt*(23.0/12.0*PV.Vorticity.tendency[i,k]  - 16.0/12.0*PV.Vorticity.now[i,k]  + 5.0/12.0*PV.Vorticity.old[i,k] )
					PV.T.spectral[i,k]          = F_T[i,k]          + self.dt*(23.0/12.0*PV.T.tendency[i,k]          - 16.0/12.0*PV.T.now[i,k]          + 5.0/12.0*PV.T.old[i,k]         )
					if Pr.moist_index > 0.0:
						PV.QT.spectral[i,k]         = F_QT[i,k]         + self.dt*(23.0/12.0*PV.QT.tendency[i,k]         - 16.0/12.0*PV.QT.now[i,k]         + 5.0/12.0*PV.QT.old[i,k]        )

		DF.update(Pr, Gr, PV, self.dt)
		self.t = self.t+self.dt
		PV.set_old_with_now()
		PV.set_now_with_tendencies()

		self.ncycle += 1
		return

	def CFL_limiter(self, Pr, Gr, DV, namelist):
		nx = Pr.nlats
		ny = Pr.nlons
		nl = Pr.n_layers

		dt = namelist['timestepping']['dt']
		CFL_limit = namelist['timestepping']['CFL_limit']
		for i in range(nx):
			for j in range(ny):
				for k in range(nl):
					dt = np.min(dt,
					CFL_limit*np.min(Gr.dx[i,j]/(np.abs(DV.U.values[i,j,k])+ 0.1) ,Gr.dy[i,j]/(np.abs(DV.V.values[i,j,k])+ 0.1)))
		return dt
