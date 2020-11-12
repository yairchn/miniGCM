from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *
from scipy.signal import savgol_filter


cdef class NumericalDiffusion:
    # cdef: this one is for init
    cpdef initialize(self, Grid Gr,TimeStepping TS, namelist)
    cpdef update(self, Grid Gr, PrognosticVariables PV, namelist, dt)