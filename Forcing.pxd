import cython
from Grid cimport Grid
from DiagnosticVariables cimport DiagnosticVariables
import numpy as np
cimport numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

cdef class ForcingBase:
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class ForcingNone(ForcingBase):
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class ForcingBettsMiller(ForcingBase):
	cdef:
		Py_ssize_t nx
		Py_ssize_t ny
		Py_ssize_t nl
		double [:,:,:] Tbar

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)