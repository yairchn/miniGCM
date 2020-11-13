import sys
import cython
from Grid cimport Grid

cdef class Thermodynamics:
	def __init__(self, namelist):
		self.cp = namelist['thermodynamics']['heat_capacity']
		self.Rd = namelist['thermodynamics']['ideal_gas_constant']
		self.kappa = self.R/self.cp
		return

	cpdef initialize(self):
		return

	cpdef update(self):
		return