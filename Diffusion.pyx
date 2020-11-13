import cython
from Grid import Grid
from math import *
import matplotlib.pyplot as plt
import numpy as np
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
from scipy.signal import savgol_filter
import time
from TimeStepping cimport TimeStepping
import sys
from sphTrans import Spharmt

cdef class Diffusion:

    def __init__(self):
        return

    cpdef initialize(self, Grid Gr, TimeStepping TS, namelist):
        dissipation_order = namelist['diffusion']['dissipation_order']
        truncation_order = namelist['diffusion']['truncation_order']
        truncation_number = int(Gr.nlons/truncation_order)
        diffusion_factor = np.multiply(1e-5,np.power(np.divide(Spharmt.lap,Spharmt.lap[-1]),(dissipation_order/2)))
        self.HyperDiffusionFactor = np.exp(-TS.dt*diffusion_factor)
        # all wave numbers above this value are removed
        self.HyperDiffusionFactor[Spharmt._shtns.l>=Gr.truncation_number] = 0.0
        return

    cpdef update(self, Grid Gr, PrognosticVariables PV, namelist, dt):
        for k in range(Gr.n_layers):
            dissipation_order = namelist['diffusion']['dissipation_order']
            diffusion_factor = np.multiply(1e-5,np.power(np.divide(Spharmt.lap,Spharmt.lap[-1]),(dissipation_order/2)))
            self.HyperDiffusionFactor = np.exp(-dt*diffusion_factor)
            self.HyperDiffusionFactor[Spharmt._shtns.l>=Gr.truncation_number] = 0.0
            PV.P.spectral[:,k]  = np.multiply(self.HyperDiffusionFactor, PV.P.spectral[:,k])
            PV.Vorticity.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Vorticity.spectral[:,k])
            PV.Divergence.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Divergence.spectral[:,k])
            PV.T.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.T.spectral[:,k])
            PV.QT.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.QT.spectral[:,k])
        return