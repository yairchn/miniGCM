import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from UtilityFunctions import axisymmetric_mean

cdef class Grid:
	def __init__(self, Parameters Pr, namelist):
		self.dx = namelist['grid']['x_resolution']
		self.dy = namelist['grid']['y_resolution']
		self.nl = namelist['grid']['number_of_layers']
		if Pr.numerical_scheme == 'weno_5':
			self.ng = 5 # 5 ghost points on each side
		if Pr.numerical_scheme == 'centeral_differneces':
			self.ng = 1 # 2 ghost points on each side
		self.nx = namelist['grid']['number_of_x_points']# + 2*self.ng
		self.ny = namelist['grid']['number_of_y_points']# + 2*self.ng

		self.x = np.zeros(self.nx+2*self.ng,dtype=np.float64, order='c')
		self.y = np.zeros(self.ny+2*self.ng,dtype=np.float64, order='c')
		for i in range(self.nx+2*self.ng):
			self.x[i] = self.dx*i
		for j in range(self.ny+2*self.ng):
			self.y[j] = self.dy*j

		self.xc = int(self.nx/2)
		self.yc = int(self.ny/2)
		X, Y = np.meshgrid(self.x, self.y)
		R = np.add(np.power(np.subtract(X,self.x[self.xc]),2.0),np.power(np.subtract(Y,self.y[self.yc]),2.0))
		self.r = axisymmetric_mean(self.xc, self.yc, R)
		self.nr = len(self.r)
		return