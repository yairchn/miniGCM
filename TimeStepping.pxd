import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Grid cimport Grid
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Diffusion cimport Diffusion
from Parameters cimport Parameters

cdef class TimeStepping:
	cdef:
		public double dt
		public double t_max
		public double t
		double ncycle
		double [:,:] dx
		double [:,:] dy

	cpdef initialize(self)
	cpdef update(self, Parameters Pr, Grid Gr,  PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist)
	cpdef CFL_limiter(self, Parameters Pr, Grid Gr, DiagnosticVariables DV, namelist)



