import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from Parameters cimport Parameters

cdef class Diffusion:
	cdef:
		double complex [:] HyperDiffusionFactor
		double complex [:] diffusion_factor
		double dissipation_order
		double truncation_order
		int truncation_number

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt)