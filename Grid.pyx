import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

cdef class Grid:
	def __init__(self, Parameters Pr, namelist):
		self.dx = namelist['grid']['x_resolution']
		self.dy = namelist['grid']['y_resolution']
		self.nl = namelist['grid']['number_of_layers']
		if Pr.numerical_scheme == 'weno_5':
			self.ng = 5 # 5 ghost points on each side
		if Pr.numerical_scheme == 'centeral_differneces':
			self.ng = 2 # 2 ghost points on each side
		self.nx = namelist['grid']['number_of_x_points'] + 2*self.ng
		self.ny = namelist['grid']['number_of_y_points'] + 2*self.ng

		self.x = np.zeros(self.nx,dtype=np.float64, order='c')
		self.y = np.zeros(self.ny,dtype=np.float64, order='c')
		for i in range(self.nx):
			self.x[i] = self.dx*i
		for j in range(self.ny):
			self.y[j] = self.dy*j
		return

