import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable

cpdef set_min_vapour(qp,qbar)
cpdef keSpectra(Grid Gr, u, v)
cdef void weno3_flux_divergence(Parameters Pr, Grid Gr, PrognosticVariable U,
                    PrognosticVariable V, PrognosticVariable Var)
cdef double interp_weno3(double phim1, double phi, double phip1) nogil
cdef double interp_weno5(double phim2, double phim1, double phi, double phip1, double phip2) nogil
cdef double roe_velocity(double fp, double fm, double varp, double varm) nogil