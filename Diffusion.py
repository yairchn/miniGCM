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
        self.diffusion_factor = (1./Gr.efold*((Gr.SphericalGrid.lap/Gr.SphericalGrid.lap[-1])**(Gr.dissipation_order/2)))
        self.HyperDiffusionFactor = np.exp(-TS.dt*self.diffusion_factor)
        # all wave numbers above this value are removed
        self.HyperDiffusionFactor[Gr.SphericalGrid._shtns.l>=Gr.truncation_number] = 0.0
        return

    def update(self, Gr, PV, namelist, dt):
        for k in range(Gr.n_layers):
            self.HyperDiffusionFactor = np.exp(-dt*self.diffusion_factor)
            self.HyperDiffusionFactor[Gr.SphericalGrid._shtns.l>=Gr.truncation_number] = 0.0
            PV.Vorticity.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Vorticity.spectral[:,k])
            PV.Divergence.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.Divergence.spectral[:,k])
            PV.T.spectral[:,k] = np.multiply(self.HyperDiffusionFactor,PV.T.spectral[:,k])

        PV.P.spectral[:,Gr.n_layers]  = np.multiply(self.HyperDiffusionFactor, PV.P.spectral[:,Gr.n_layers])
        return
