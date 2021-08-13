import cython
from Grid cimport Grid
from DiagnosticVariables cimport DiagnosticVariables
import numpy as np
cimport numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
import sphericalForcing as spf
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters
import sphericalForcing as spf

cdef class ForcingBase:
	cdef:
		double [:,:,:] Tbar
		double [:,:] sin_lat
		double [:,:] cos_lat
		double complex [:] F0 
		double [:,:] forcing_noise 
		bint noise

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class ForcingNone(ForcingBase):
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class HelzSuarez(ForcingBase):
	cdef:
		Py_ssize_t nx
		Py_ssize_t ny
		Py_ssize_t nl

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class StochasticTropicalPlanet(ForcingBase):
	cdef:
		Py_ssize_t nx
		Py_ssize_t ny
		Py_ssize_t nl
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)

cdef class StochasticHeldSuarez(ForcingBase):
	cdef:
		Py_ssize_t nx
		Py_ssize_t ny
		Py_ssize_t nl
	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef initialize_io(self, NetCDFIO_Stats Stats)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
	cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats)
	cpdef stats_io(self, NetCDFIO_Stats Stats)
