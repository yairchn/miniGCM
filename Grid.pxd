import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

cdef class Grid:
	cdef:
		double [:] x
		double [:] y
		double Coriolis
		double dx
		double dy