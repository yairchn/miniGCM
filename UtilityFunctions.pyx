import os
from Grid cimport Grid
import numpy as np
import scipy as sc
from math import *
import sys
import cython
import Parameters

# make sure that total moisture content is non-negative
cpdef set_min_vapour(qp,qbar):
    qtot = qp + qbar
    qtot[qtot<0] = 0
    return (qtot-qbar)

# Function for plotting KE spectra
cpdef keSpectra(u,v):
    uk = x.grdtospec(u)
    vk = x.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
    return [Ek,k]