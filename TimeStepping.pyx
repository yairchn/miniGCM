#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from libc.math cimport fmax, fmin, fabs, floor
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
			Py_ssize_t nx = Gr.nx
			Py_ssize_t ny = Gr.ny
			Py_ssize_t nl = Pr.n_layers
			double [:,:,:] F_V = np.zeros((nx,ny,nl), dtype = np.float64, order='c')
			double [:,:,:] F_U = np.zeros((nx,ny,nl), dtype = np.float64, order='c')
			double [:,:,:] F_T = np.zeros((nx,ny,nl), dtype = np.float64, order='c')
			double [:,:,:] F_QT= np.zeros((nx,ny,nl), dtype = np.float64, order='c')
			double [:,:]   F_P = np.zeros((nx,ny),    dtype = np.float64, order='c')

		# self.CFL_limiter(Pr, Gr, DV, namelist)
		dt = namelist['timestepping']['dt']

		F_V  = PV.V.values
		F_U  = PV.U.values
		F_T  = PV.T.values
		F_QT = PV.QT.values
		F_P  = PV.P.values[:,:,nl]

		if self.ncycle==0:
			PV.set_now_with_tendencies()
			with nogil:
				for i in range(nx):
					for j in range(ny):
						for k in range(nl):
							# Euler
							# new                   old                      tendency
							PV.V.values[i,j,k]  = F_V[i,j,k]  + self.dt*PV.V.tendency[i,j,k]
							PV.U.values[i,j,k]  = F_U[i,j,k]  + self.dt*PV.U.tendency[i,j,k]
							PV.T.values[i,j,k]  = F_T[i,j,k]  + self.dt*PV.T.tendency[i,j,k]
							PV.QT.values[i,j,k] = F_QT[i,j,k] + self.dt*PV.QT.tendency[i,j,k]
						PV.P.values[i,j,nl]       = F_P[i,j]    + self.dt*PV.P.tendency[i,j,nl]
		elif self.ncycle==1:
			with nogil:
				for i in range(nx):
					for j in range(ny):
						for k in range(nl):
							#2nd order AB
							#           new        old                        tendency                tendency now
							PV.V.values[i,j,k]  = F_V[i,j,k]  + self.dt*(1.5*PV.V.tendency[i,j,k]  - 0.5*PV.V.now[i,j,k])
							PV.U.values[i,j,k]  = F_U[i,j,k]  + self.dt*(1.5*PV.U.tendency[i,j,k]  - 0.5*PV.U.now[i,j,k])
							PV.T.values[i,j,k]  = F_T[i,j,k]  + self.dt*(1.5*PV.T.tendency[i,j,k]  - 0.5*PV.T.now[i,j,k])
							PV.QT.values[i,j,k] = F_QT[i,j,k] + self.dt*(1.5*PV.QT.tendency[i,j,k] - 0.5*PV.QT.now[i,j,k])
						PV.P.values[i,j,nl]     = F_P[i,j]    + self.dt*(1.5*PV.P.tendency[i,j,nl] - 0.5*PV.P.now[i,j,nl])
		else:
			with nogil:
				for i in range(nx):
					for j in range(ny):
						for k in range(nl):
							#3nd order AB
							#   new                old                               tendency                      tendency now              tendency old
							PV.V.values[i,j,k]  = F_V[i,j,k]  + self.dt*(23.0/12.0*PV.V.tendency[i,j,k]  - 16.0/12.0*PV.V.now[i,j,k]  + 5.0/12.0*PV.V.old[i,j,k] )
							PV.U.values[i,j,k]  = F_U[i,j,k]  + self.dt*(23.0/12.0*PV.U.tendency[i,j,k]  - 16.0/12.0*PV.U.now[i,j,k]  + 5.0/12.0*PV.U.old[i,j,k] )
							PV.T.values[i,j,k]  = F_T[i,j,k]  + self.dt*(23.0/12.0*PV.T.tendency[i,j,k]  - 16.0/12.0*PV.T.now[i,j,k]  + 5.0/12.0*PV.T.old[i,j,k] )
							PV.QT.values[i,j,k] = F_QT[i,j,k] + self.dt*(23.0/12.0*PV.QT.tendency[i,j,k] - 16.0/12.0*PV.QT.now[i,j,k] + 5.0/12.0*PV.QT.old[i,j,k])
						PV.P.values[i,j,nl]     = F_P[i,j]    + self.dt*(23.0/12.0*PV.P.tendency[i,j,nl] - 16.0/12.0*PV.P.now[i,j,nl] + 5.0/12.0*PV.P.old[i,j,nl])

		DF.update(Pr, Gr, PV, self.dt)
		self.t = self.t+self.dt
		PV.set_old_with_now()
		PV.set_now_with_tendencies()

		self.ncycle += 1
		return

	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.nonecheck(False)
	cpdef CFL_limiter(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
		cdef:
			Py_ssize_t i, j, k
			Py_ssize_t nx = Pr.nx
			Py_ssize_t ny = Pr.ny
			Py_ssize_t nl = Pr.n_layers
			double CFL_limit, dt

		dt = namelist['timestepping']['dt']
		CFL_limit = namelist['timestepping']['CFL_limit']
		with nogil:
			for i in range(nx):
				for j in range(ny):
					for k in range(nl):
						dt = fmin(dt,
						CFL_limit*fmin(Gr.dx/(fabs(PV.U.values[i,j,k])+ 0.1) ,Gr.dy/(fabs(PV.V.values[i,j,k])+ 0.1)))
		return dt
