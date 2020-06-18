from __future__ import print_function
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
# import AdamsBashforth
import sphericalForcing as spf
import scipy as sc
import xarray
import logData
import netCDF4
import seaborn
import numpy as np
from math import *
from scipy.signal import savgol_filter


class NumericalDiffusion:

    def __init__(self):
        return

    def initialize(self, Gr, TS, namelist):
        dissipation_order = namelist['diffusion']['dissipation_order']
        truncation_order = namelist['diffusion']['truncation_order']
        diffusion_factor = (1e-5*((Gr.SphericalGrid.lap/Gr.SphericalGrid.lap[-1])**(dissipation_order/2)))
        self.HyperDiffusionFactor = np.exp(-TS.dt*diffusion_factor)
        # all wave numbers above this value are removed
        self.HyperDiffusionFactor[Gr.SphericalGrid._shtns.l>=Gr.truncation_number] = 0.0
        return

    def update(self, Gr, PV):
        for k in range(Gr.n_layers):
            PV.P.spectral[:,k]  = np.multiply(PV.P.spectral[:,k] ,self.HyperDiffusionFactor)
            PV.Vorticity.spectral[:,k] *= self.HyperDiffusionFactor
            PV.Divergence.spectral[:,k] *= self.HyperDiffusionFactor
            PV.T.spectral[:,k] *= self.HyperDiffusionFactor
            PV.QT.spectral[:,k] *= self.HyperDiffusionFactor
        return