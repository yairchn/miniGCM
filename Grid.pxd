import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

cdef class Grid:
	cdef:
		int nx
		int ny
		int nr
		int nl
		int ng
		double dx
		double dy
		int xc
		int yc
		double [:] x
		double [:] y
		double [:] r
		double Coriolis