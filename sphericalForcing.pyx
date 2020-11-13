import cython
from Grid cimport Grid
from math import *
import netCDF4
import numpy as np
from scipy.signal import savgol_filter
import scipy as sc
import shtns
import sphTrans
import sphericalForcing as spf
import time
import sys
import xarray

cdef class sphForcing(object):
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

    cpdef forcingFn(self, double [:,:,:] F0):
        signal = self.magnitude* self.A* np.exp(np.random.rand(self.nlm)*1j*2*np.pi)
        F = np.add((np.sqrt(1-self.corr**2))*signal,np.multiply(self.corr,F0))
        out = F
        out[self.m==0] = 0. # Remove zonal mean component
        return out

    cpdef update(self,double [:,:,:] F0):
        signal = self.magnitude* self.A* np.exp(np.random.rand(self.nlm)*1j*2*np.pi)
        F = np.add((np.sqrt(1-self.corr**2))*signal,np.multiply(self.corr,F0))
        out = F
        out[self.m==0] = 0. # Remove zonal mean component
        return out
