import cython
from Grid cimport Grid
from math import *
import netCDF4
import numpy as np
import scipy as sc
from scipy.signal import savgol_filter
import shtns
import sphTrans
import sphericalForcing as spf
import time
import sys
import xarray

cdef class sphForcing:
    cdef:
        double lmin
        double lmax
        double corr
        double magnitude
        double ntrunc
        double rsphere
        double trans
        double l
        double m
        double A
        double nlm

    cpdef forcingFn(self, double [:,:,:] F0)
    cpdef update(self,double [:,:,:] F0)
