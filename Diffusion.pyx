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

cdef class Diffusion:

    def __init__(self):
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        self.diffusion_factor = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
        self.HyperDiffusionFactor = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
        return

    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt):
        for i in range (Gr.SphericalGrid.nlm):
            self.diffusion_factor[i] = (1.0/Pr.efold*((Gr.SphericalGrid.lap[i]/Gr.SphericalGrid.lap[-1])**(Pr.dissipation_order/2)))
            self.HyperDiffusionFactor[i] = np.exp(-dt*self.diffusion_factor[i])
        self.HyperDiffusionFactor.base[Gr.SphericalGrid._shtns.l>=Pr.truncation_number] = 0.0
        for k in range(Pr.n_layers):
            PV.P.spectral.base[:,k]  = np.multiply(self.HyperDiffusionFactor, PV.P.spectral[:,k])
            PV.Vorticity.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Vorticity.spectral[:,k])
            PV.Divergence.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Divergence.spectral[:,k])
            PV.T.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.T.spectral[:,k])
            PV.QT.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.QT.spectral[:,k])
        return

    # cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
    #     return
    # # @cython.wraparound(False)
    # @cython.boundscheck(False)
    # cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt):
    #     cdef:
    #         Py_ssize_t i,j,k
    #         Py_ssize_t nl = Pr.n_layers
    #         Py_ssize_t nlm = Gr.SphericalGrid.nlm
    #         int [:] shtns_l
    #         double complex diffusion_factor
    #         double complex HyperDiffusionFactor

    #     shtns_l = np.copy(Gr.SphericalGrid._shtns.l)

    #     with nogil:
    #         for i in range(nlm):
    #             diffusion_factor = (1.0/Pr.efold*((Gr.SphericalGrid.lap[i]/Gr.SphericalGrid.lap[-1])**(Pr.dissipation_order/2)))
    #             if shtns_l[i]>=Pr.truncation_number:
    #                 HyperDiffusionFactor = exp(-dt*diffusion_factor.real)
    #             else:
    #                 HyperDiffusionFactor =0.0
    #             for k in range(nl):
    #                 PV.P.spectral[i,k]          = HyperDiffusionFactor * PV.P.spectral[i,k]
    #                 PV.Vorticity.spectral[i,k]  = HyperDiffusionFactor * PV.Vorticity.spectral[i,k]
    #                 PV.Divergence.spectral[i,k] = HyperDiffusionFactor * PV.Divergence.spectral[i,k]
    #                 PV.T.spectral[i,k]          = HyperDiffusionFactor * PV.T.spectral[i,k]
    #                 PV.QT.spectral[i,k]         = HyperDiffusionFactor * PV.QT.spectral[i,k]
    #     return
