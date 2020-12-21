import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import numpy as np
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
from scipy.signal import savgol_filter
import time
from TimeStepping cimport TimeStepping
import sys

cdef class Diffusion:

    def __init__(self):
        return

    cpdef initialize(self, Grid Gr, TimeStepping TS, namelist):
        self.dissipation_order = namelist['diffusion']['dissipation_order']
        self.truncation_order = namelist['diffusion']['truncation_order']
        self.truncation_number = int(Gr.nlons/truncation_order)
        self.diffusion_factor = (1e-5*((Gr.SphericalGrid.lap/Gr.SphericalGrid.lap[-1])**(dissipation_order/2)))
        self.HyperDiffusionFactor = np.exp(-TS.dt*diffusion_factor)
        # all wave numbers above this value are removed
        self.HyperDiffusionFactor[Gr.SphericalGrid._shtns.l>=Gr.truncation_number] = 0.0
        return

    cpdef update(self, Grid Gr, PrognosticVariables PV, double dt):
        for k in range(Gr.n_layers):
            self.HyperDiffusionFactor = np.exp(-dt*self.diffusion_factor)
            self.HyperDiffusionFactor[Gr.SphericalGrid._shtns.l>=Gr.truncation_number] = 0.0
            PV.P.spectral[:,k]  = np.multiply(self.HyperDiffusionFactor, PV.P.spectral[:,k])
            PV.Vorticity.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Vorticity.spectral[:,k])
            PV.Divergence.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Divergence.spectral[:,k])
            PV.T.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.T.spectral[:,k])
            PV.QT.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.QT.spectral[:,k])
        return