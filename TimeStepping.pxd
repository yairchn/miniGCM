import numpy as np
import sys
import cython
from Grid cimport Grid
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Diffusion cimport Diffusion

cdef class TimeStepping:
	cdef:
		public double dt
		public double t_max
		public double t
		double ncycle
		double dx
		double dy
		double dp
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist)
	cpdef update(self, Grid Gr,  PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist)



