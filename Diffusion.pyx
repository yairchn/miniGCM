import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from Parameters cimport Parameters
from libc.math cimport exp

# overload exp for complex numbers
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
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm
            int [:] shtns_l
            double complex diffusion_factor_meso_scale
            double complex diffusion_factor_grid_scale
            double complex HyperDiffusionFactor
            double complex [:] laplacian

        shtns_l = np.copy(Gr.SphericalGrid._shtns.l)
        laplacian  = Gr.SphericalGrid.lap

        with nogil:
            for i in range(nlm):
                diffusion_factor_meso_scale = (1.0/Pr.efold*((laplacian[i]/laplacian[-1])**(Pr.dissipation_order/2.0)))
                #diffusion_factor_grid_scale= (1.0/Pr.efold*((laplacian[i]/laplacian[-1])**(Pr.dissipation_order_grid_scale/2.0)))
                HyperDiffusionFactor = exp(-dt*(diffusion_factor_meso_scale))#+diffusion_factor_grid_scale))
                if shtns_l[i]>=Pr.truncation_number:
                    HyperDiffusionFactor = 0.0
                for k in range(nl):
                    PV.P.spectral[i,k]          = HyperDiffusionFactor * PV.P.spectral[i,k]
                    PV.Vorticity.spectral[i,k]  = HyperDiffusionFactor * PV.Vorticity.spectral[i,k]
                    PV.Divergence.spectral[i,k] = HyperDiffusionFactor * PV.Divergence.spectral[i,k]
                    PV.T.spectral[i,k]          = HyperDiffusionFactor * PV.T.spectral[i,k]
                    PV.QT.spectral[i,k]         = HyperDiffusionFactor * PV.QT.spectral[i,k]
        return
