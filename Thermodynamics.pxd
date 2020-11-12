import scipy as sc
import numpy as np
from math import *

cdef class Thermodynamics:
	cdef:
		double cp
		double Rd
		double kappa
	cpdef initialize(self)
	cpdef update(self)