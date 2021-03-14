import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from Parameters cimport Parameters
from libc.math cimport exp

# overload exp for complex number
cdef extern from "complex.h" nogil:
    double complex exp(double complex z)

cdef class Diffusion:

    def __init__(self):
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        return
    #@cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t ng = Gr.ng
            Py_ssize_t nx = Gr.nx
            Py_ssize_t ny = Gr.ny
            Py_ssize_t nl = Gr.nl
            double diffusion_factor
            double HyperDiffusionFactor

        with nogil:
            for i in range(nx):
                for j in range(ny):
                    diffusion_factor = (1.0/Pr.efold*((laplacian[i]/laplacian[-1])**(Pr.dissipation_order/2.0)))
                    HyperDiffusionFactor = 1.0
                    for k in range(nl):
                        PV.P.values[i,j,k]  = HyperDiffusionFactor * (PV.P.values[i,j,k])
                        PV.U.values[i,j,k]  = HyperDiffusionFactor * (PV.U.values[i,j,k])
                        PV.V.values[i,j,k]  = HyperDiffusionFactor * (PV.V.values[i,j,k])
                        PV.T.values[i,j,k]  = HyperDiffusionFactor * (PV.T.values[i,j,k])
                        PV.QT.values[i,j,k] = HyperDiffusionFactor * (PV.QT.values[i,j,k])
        return
