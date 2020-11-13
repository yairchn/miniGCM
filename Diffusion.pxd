import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import numpy as np
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
from scipy.signal import savgol_filter
import time
from TimeStepping cimport TimeStepping
import sys

cdef class Diffusion:
    # cdef: this one is for init
    cpdef initialize(self, Grid Gr,TimeStepping TS, namelist)
    cpdef update(self, Grid Gr, PrognosticVariables PV, namelist, dt)