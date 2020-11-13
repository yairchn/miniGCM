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

cdef class Spharmt(object):

    cdef:
        double [:,:] _shtns
        double lats
        double lons
        double nlons
        double nlats
        double ntrunc
        double nlm
        double degree
        double lap
        double rsphere
        double invlap

    cpdef grdtospec(self, data)
    cpdef spectogrd(self,dataspec)
    cpdef getuv(self,vrtspec,divspec)
    cpdef getvrtdivspec(self,u,v)
    cpdef getgrad(self,divspec)