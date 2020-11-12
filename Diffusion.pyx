from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *
from scipy.signal import savgol_filter


class NumericalDiffusion:

    def __init__(self):
        return

    def initialize(self, Gr, TS, namelist):
        dissipation_order = namelist['diffusion']['dissipation_order']
        truncation_order = namelist['diffusion']['truncation_order']
        truncation_number = int(Gr.nlons/truncation_order)
        diffusion_factor = (1e-5*((Gr.SphericalGrid.lap/Gr.SphericalGrid.lap[-1])**(dissipation_order/2)))
        self.HyperDiffusionFactor = np.exp(-TS.dt*diffusion_factor)
        # all wave numbers above this value are removed
        self.HyperDiffusionFactor[Gr.SphericalGrid._shtns.l>=Gr.truncation_number] = 0.0
        return

    def update(self, Gr, PV, namelist, dt):
        for k in range(Gr.n_layers):
            dissipation_order = namelist['diffusion']['dissipation_order']
            diffusion_factor = (1e-5*((Gr.SphericalGrid.lap/Gr.SphericalGrid.lap[-1])**(dissipation_order/2)))
            self.HyperDiffusionFactor = np.exp(-dt*diffusion_factor)
            self.HyperDiffusionFactor[Gr.SphericalGrid._shtns.l>=Gr.truncation_number] = 0.0
            PV.P.spectral[:,k]  = np.multiply(self.HyperDiffusionFactor, PV.P.spectral[:,k])
            PV.Vorticity.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Vorticity.spectral[:,k])
            PV.Divergence.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Divergence.spectral[:,k])
            PV.T.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.T.spectral[:,k])
            PV.QT.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.QT.spectral[:,k])
        return