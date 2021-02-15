import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

cdef class Grid:
	def __init__(self, Parameters Pr, namelist):
		self.dx = namelist['grid']['dx']
		self.dy = namelist['grid']['dy']
		self.x = np.zeros((Pr.nx),dtype=np.float64, order='c')
		self.y = np.zeros((Pr.ny),dtype=np.float64, order='c')
		for i in range(nx):
			self.x[i] = self.dx*i
		for j in range(nx):
			self.y[j] = self.dy*j
		return

