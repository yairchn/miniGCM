import numpy as np
import scipy as sc
from math import *

cdef class SurfaceNone:
    cdef:
    cpdef initialize(self)
    cpdef update(self)
    cpdef initialize_io(self, Stats)
    cpdef io(self, Stats)

cdef class Surface_HelzSuarez:
	cdef:
	cpdef initialize(self, Gr, Stats)
	cpdef update()