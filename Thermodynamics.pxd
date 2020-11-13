import sys
import cython
from Grid import Grid

cdef class Thermodynamics:
	cdef:
		double cp
		double Rd
		double kappa
	cpdef initialize(self)
	cpdef update(self)