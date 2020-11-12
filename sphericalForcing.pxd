from __future__ import print_function
import numpy as np
import sphTrans
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import sphericalForcing as spf
import scipy as sc
import xarray
import logData
import netCDF4
import numpy as np
from math import *
from scipy.signal import savgol_filter

cdef class sphForcing:
    cdef:
        double [:] lmin
        double [:] lmax
        double [:] corr
        double [:] magnitude
        double [:] ntrunc
        double [:] rsphere
        double [:] trans
        double [:] l
        double [:] m
        double [:] A
        double [:] nlm

    cpdef forcingFn(self, double [:,:,:] F0)
    cpdef update(self,double [:,:,:] F0)
