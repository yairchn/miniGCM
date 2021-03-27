#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from libc.math cimport fmax, fmin, fabs, floor, sqrt
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

	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	@cython.nonecheck(False)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist):
		cdef:
			Py_ssize_t i, k
			Py_ssize_t nl = Pr.n_layers
			Py_ssize_t nlm = Gr.SphericalGrid.nlm
			double complex [:,:] F_Divergence = np.zeros((nlm, nl), dtype = np.complex, order='c')
			double complex [:,:] F_Vorticity  = np.zeros((nlm, nl), dtype = np.complex, order='c')
			double complex [:,:] F_H          = np.zeros((nlm, nl), dtype = np.complex, order='c')
			double complex [:,:] F_QT         = np.zeros((nlm, nl), dtype = np.complex, order='c')
			double complex [:]   F_P          = np.zeros((nlm),     dtype = np.complex, order='c')

		# self.CFL_limiter(Pr, Gr, DV, namelist)
		dt = namelist['timestepping']['dt']

		F_Divergence = PV.Divergence.spectral
		F_Vorticity  = PV.Vorticity.spectral
		F_H          = PV.H.spectral
		F_QT         = PV.QT.spectral

		if self.ncycle==0:
			PV.set_now_with_tendencies()
			with nogil:
				for i in range(nlm):
					for k in range(nl):
						# Euler
						# new                                old                      tendency
						PV.Divergence.spectral[i,k] = F_Divergence[i,k] + self.dt*PV.Divergence.tendency[i,k]
						PV.Vorticity.spectral[i,k]  = F_Vorticity[i,k]  + self.dt*PV.Vorticity.tendency[i,k]
						PV.H.spectral[i,k]          = F_H[i,k]          + self.dt*PV.H.tendency[i,k]
						PV.QT.spectral[i,k]         = F_QT[i,k]         + self.dt*PV.QT.tendency[i,k]
		elif self.ncycle==1:
			with nogil:
				for i in range(nlm):
					for k in range(nl):
						#2nd order AB
						#           new                     old                        tendency                            tendency now
						PV.Divergence.spectral[i,k] = F_Divergence[i,k] + self.dt*(1.5*PV.Divergence.tendency[i,k] - 0.5*PV.Divergence.now[i,k])
						PV.Vorticity.spectral[i,k]  = F_Vorticity[i,k]  + self.dt*(1.5*PV.Vorticity.tendency[i,k]  - 0.5*PV.Vorticity.now[i,k])
						PV.H.spectral[i,k]          = F_H[i,k]          + self.dt*(1.5*PV.H.tendency[i,k]          - 0.5*PV.H.now[i,k])
						PV.QT.spectral[i,k]         = F_QT[i,k]         + self.dt*(1.5*PV.QT.tendency[i,k]         - 0.5*PV.QT.now[i,k])
		else:
			with nogil:
				for i in range(nlm):
					for k in range(nl):
						#3nd order AB
						#           new                    old                               tendency                                tendency now                         tendency old
						PV.Divergence.spectral[i,k] = F_Divergence[i,k] + self.dt*(23.0/12.0*PV.Divergence.tendency[i,k] - 16.0/12.0*PV.Divergence.now[i,k] + 5.0/12.0*PV.Divergence.old[i,k])
						PV.Vorticity.spectral[i,k]  = F_Vorticity[i,k]  + self.dt*(23.0/12.0*PV.Vorticity.tendency[i,k]  - 16.0/12.0*PV.Vorticity.now[i,k]  + 5.0/12.0*PV.Vorticity.old[i,k] )
						PV.H.spectral[i,k]          = F_H[i,k]          + self.dt*(23.0/12.0*PV.H.tendency[i,k]          - 16.0/12.0*PV.H.now[i,k]          + 5.0/12.0*PV.H.old[i,k]         )
						PV.QT.spectral[i,k]         = F_QT[i,k]         + self.dt*(23.0/12.0*PV.QT.tendency[i,k]         - 16.0/12.0*PV.QT.now[i,k]         + 5.0/12.0*PV.QT.old[i,k]        )

		DF.update(Pr, Gr, PV, self.dt)
		self.t = self.t+self.dt
		PV.set_old_with_now()
		PV.set_now_with_tendencies()

		self.ncycle += 1
		return

	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.nonecheck(False)
	cpdef CFL_limiter(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
		cdef:
			Py_ssize_t i, j, k
			Py_ssize_t nx = Pr.nlats
			Py_ssize_t ny = Pr.nlons
			Py_ssize_t nl = Pr.n_layers
			double CFL_limit, dt, U, V, gravitywave

		dt = namelist['timestepping']['dt']
		CFL_limit = namelist['timestepping']['CFL_limit']
		with nogil:
			for i in range(nx):
				for j in range(ny):
					for k in range(nl):
						gravitywave = sqrt(Pr.g*PV.H.values[i,j,k])
						U = fmax(fabs(DV.U.values[i,j,k]) + 0.1, gravitywave)
						V = fmax(fabs(DV.V.values[i,j,k]) + 0.1, gravitywave)
						dt = fmin(dt,
						CFL_limit*fmin(Gr.dx[i,j]/U ,Gr.dy[i,j]/V))
		return dt
