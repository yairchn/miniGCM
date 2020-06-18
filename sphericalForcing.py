from __future__ import print_function
import numpy as np
import sphTrans
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

class sphForcing(object):
    def __init__(self, nlons, nlats, ntrunc, rsphere, lmin, lmax, magnitude , correlation = 0.5):
        self.lmin = lmin
        self.lmax = lmax
        self.corr = correlation
        self.magnitude = magnitude
        self.ntrunc = ntrunc
        self.rsphere = rsphere
        self.trans = sphTrans.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
        self.l = self.trans._shtns.l
        self.m = self.trans._shtns.m
        A = np.zeros(self.trans.nlm)
        A[self.l >= self.lmin] = 1.
        A[self.l >  self.lmax] = 0.
        A[self.m == 0] = 0.     # Removing zonal mean
        self.A = A
        self.nlm = self.trans._shtns.nlm

    def forcingFn(self,F0):
        signal = self.magnitude* self.A* np.exp(np.random.rand(self.nlm)*1j*2*np.pi)
        F = (np.sqrt(1-self.corr**2))*signal + self.corr*F0
        out = F
        out[self.m==0] = 0. # Remove zonal mean component
        return out

    def update(self,F0):
        signal = self.magnitude* self.A* np.exp(np.random.rand(self.nlm)*1j*2*np.pi)
        F = (np.sqrt(1-self.corr**2))*signal + self.corr*F0
        out = F
        out[self.m==0] = 0. # Remove zonal mean component
        return out
