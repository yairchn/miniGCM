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

cdef class Diffusion:

    def __init__(self):
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        Pr.dissipation_order = namelist['diffusion']['dissipation_order']
        Pr.truncation_order = namelist['diffusion']['truncation_order']
        Pr.truncation_number = int(Pr.nlons/Pr.truncation_order)
        self.diffusion_factor = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
        self.HyperDiffusionFactor = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
        return

    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt):
        for i in range (Gr.SphericalGrid.nlm):
            self.diffusion_factor[i] = (1e-5*((Gr.SphericalGrid.lap[i]/Gr.SphericalGrid.lap[-1])**(Pr.dissipation_order/2)))
            self.HyperDiffusionFactor[i] = np.exp(-dt*self.diffusion_factor[i])
        self.HyperDiffusionFactor.base[Gr.SphericalGrid._shtns.l>=Pr.truncation_number] = 0.0
        for k in range(Gr.n_layers):
            PV.P.spectral.base[:,k]  = np.multiply(self.HyperDiffusionFactor, PV.P.spectral[:,k])
            PV.Vorticity.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Vorticity.spectral[:,k])
            PV.Divergence.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Divergence.spectral[:,k])
            PV.T.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.T.spectral[:,k])
            PV.QT.spectral.base[:,k] = np.multiply(self.HyperDiffusionFactor,PV.QT.spectral[:,k])
        return