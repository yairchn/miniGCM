from __future__ import print_function
import numpy as np
import shtns
import numpy as np
import sphTrans as sph
import matplotlib.pyplot as plt
import time
# import AdamsBashforth
import sphericalForcing as spf
import scipy as sc
import xarray
import logData
import netCDF4
import numpy as np
from math import *
from scipy.signal import savgol_filter


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
        double invlap
        double invlap
        double rsphere
        double lap
        double invlap

    cpdef grdtospec(self, data)
    cpdef spectogrd(self,dataspec)
    cpdef getuv(self,vrtspec,divspec)
    cpdef getvrtdivspec(self,u,v)
    cpdef getgrad(self,divspec)