import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

cdef class Grid:
	def __init__(self, Parameters Pr, namelist):
		self.dx = namelist['grid']['dx']
		self.dy = namelist['grid']['dy']
		self.nx = namelist['grid']['nx']
		self.ny = namelist['grid']['ny']
		if Pr.numerical_scheme == 'weno_5':
			self.ng = 5 # 5 ghost points on each side
		if Pr.numerical_scheme == 'centeral_differneces':
			self.ng = 2 # 2 ghost points on each side

		self.x = np.zeros((self.nx+2*self.ng),dtype=np.float64, order='c')
		self.y = np.zeros((self.ny+2*self.ng),dtype=np.float64, order='c')
		for i in range(nx+2*self.ng):
			self.x[i] = self.dx*i
		for j in range(ny+2*self.ng):
			self.y[j] = self.dy*j
		return

