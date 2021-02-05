import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import numpy as np
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
from scipy.signal import savgol_filter
import sys
from Parameters cimport Parameters
from libc.math cimport exp

# overload exp for complex number
cdef extern from "complex.h" nogil:
    double complex exp(double complex z)

# cdef extern from "diffusion.h":
#     void hyperdiffusion(double dt, double efold, double dissipation_order,
#            int truncation_number, int* shtns_l,
#            double complex * laplacian, double complex * variable,
#            Py_ssize_t imax, Py_ssize_t Py_kmax) nogil

cdef class Diffusion:

    def __init__(self):
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        return
    # @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm
            int [:] shtns_l
            double complex diffusion_factor
            double complex HyperDiffusionFactor
            double complex [:] laplacian

        shtns_l = np.copy(Gr.SphericalGrid._shtns.l)
        laplacian  = Gr.SphericalGrid.lap

        with nogil:
            # hyperdiffusion(dt, Pr.efold, Pr.dissipation_order, Pr.truncation_number,
            #                 &shtns_l[0], &laplacian[0], &PV.P.spectral[0,0], nlm, nl)

            for i in range(nlm):
                diffusion_factor = (1.0/Pr.efold*((laplacian[i]/laplacian[-1])**(Pr.dissipation_order/2.0)))
                HyperDiffusionFactor = exp(-dt*diffusion_factor)
                if shtns_l[i]>=Pr.truncation_number:
                    HyperDiffusionFactor = 0.0
                for k in range(nl):
                    PV.P.spectral[i,k]          = HyperDiffusionFactor * PV.P.spectral[i,k]
                    PV.Vorticity.spectral[i,k]  = HyperDiffusionFactor * PV.Vorticity.spectral[i,k]
                    PV.Divergence.spectral[i,k] = HyperDiffusionFactor * PV.Divergence.spectral[i,k]
                    PV.T.spectral[i,k]          = HyperDiffusionFactor * PV.T.spectral[i,k]
                    PV.QT.spectral[i,k]         = HyperDiffusionFactor * PV.QT.spectral[i,k]
        return
