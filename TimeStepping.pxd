import numpy as np

cdef class TimeStepping:
	cdef:
		double dt
		double t_max
		double t
		double ncycle
		double dx
		double dy
		double dp
	cpdef initialize(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist)
	cpdef update(self, Grid Gr,  PrognosticVariables PV, DiagnosticVariables DV, Diffusion DF, namelist)



