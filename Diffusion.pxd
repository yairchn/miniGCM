import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from Parameters cimport Parameters

cdef class Diffusion:
	cdef:
		Py_ssize_t i,j,k
		Py_ssize_t ng
		Py_ssize_t nx
		Py_ssize_t ny
		Py_ssize_t nl
		double dxi_4
		double dyi_4


	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt)