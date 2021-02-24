import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from PrognosticVariables cimport PrognosticVariable

cpdef set_min_vapour(qp,qbar)
cpdef keSpectra(Grid Gr, u, v)
cdef weno3_flux_divergence(Parameters Pr, Grid Gr, double *U, double *V, double *Var)
cdef double interp_weno3(double phim1, double phi, double phip1) nogil
cdef double roe_velocity(double fp, double fm, double varp, double varm) nogil