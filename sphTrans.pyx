import cython
from Grid cimport Grid
from math import *
import netCDF4
import numpy as np
import scipy as sc
from scipy.signal import savgol_filter
import shtns
import sphericalForcing as spf
import sys
import time
import xarray


cdef class Spharmt: # (object)
    """
    wrapper class for commonly used spectral transform operations in
    atmospheric models.  Provides an interface to shtns compatible
    with pyspharm (pyspharm.googlecode.com).
    """
    def __init__(self,nlons,nlats,ntrunc,rsphere, gridtype='gaussian'):
        """initialize
        nlons:  number of longitudes
        nlats:  number of latitudes"""
        self._shtns = shtns.sht(ntrunc, ntrunc, 1, \
                shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
        if gridtype == 'gaussian':
            #self._shtns.set_grid(nlats,nlons,shtns.sht_gauss_fly|shtns.SHT_PHI_CONTIGUOUS,1.e-10)
            self._shtns.set_grid(nlats,nlons,shtns.sht_quick_init|shtns.SHT_PHI_CONTIGUOUS,1.e-10)
        elif gridtype == 'regular':
            self._shtns.set_grid(nlats,nlons,shtns.sht_reg_dct|shtns.SHT_PHI_CONTIGUOUS,1.e-10)
        self.lats = np.arcsin(self._shtns.cos_theta)
        self.lons = (2.*np.pi/nlons)*np.arange(nlons)
        self.nlons = nlons
        self.nlats = nlats
        self.ntrunc = ntrunc
        self.nlm = self._shtns.nlm
        # print(np.shape(self._shtns.l))
        # print(self._shtns.l)
        # print(len(self._shtns.l))
        self.degree = np.zeros(len(self._shtns.l),dtype=np.double, order='c')
        self.lap = np.zeros(len(self._shtns.l),dtype=np.cdouble, order='c')
        # self.invlap = np.zeros(len(self._shtns.l),dtype=np.cdouble, order='c')
        for i in range(len(self._shtns.l)):
            self.degree[i] = self._shtns.l[i]
        self.lap = -np.multiply(self.degree,(np.add(self.degree,1.0)).astype(np.complex))
        self.invlap = np.zeros_like(self.lap)
        for i in range(len(self.invlap[1:])):
            self.invlap[i+1] = np.divide(1.0,self.lap[i+1])
        # self.invlap[1:] = np.add(self.invlap[1:],np.divide(1.0,self.lap[1:]))
        self.rsphere = rsphere
        self.lap = np.divide(self.lap,self.rsphere**2)
        self.invlap = np.multiply(self.invlap,self.rsphere**2)

    cpdef grdtospec(self, data):
        """compute spectral coefficients from gridded data"""
        return self._shtns.analys(data)
    
    cpdef spectogrd(self,dataspec):
        """compute gridded data from spectral coefficients"""
        return self._shtns.synth(dataspec)
    
    cpdef getuv(self,vrtspec,divspec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""
        return self._shtns.synth(np.divide(self.invlap,self.rsphere)*vrtspec,
                np.divide(self.invlap,self.rsphere)*divspec)

    cpdef getvrtdivspec(self,u,v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        vrtspec, divspec = self._shtns.analys(u, v)
        return np.multiply(self.lap,self.rsphere*vrtspec), np.multiply(self.lap,self.rsphere*divspec)
    
    cpdef getgrad(self,divspec):
        """compute gradient vector from spectral coeffs"""
        vrtspec = np.zeros(divspec.shape, dtype=np.complex)
        u,v = self._shtns.synth(vrtspec,divspec)
        return u/self.rsphere, v/self.rsphere

