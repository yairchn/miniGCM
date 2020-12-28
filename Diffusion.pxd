import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import numpy as np
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
from scipy.signal import savgol_filter
import sys
from Parameters cimport Parameters

cdef class Diffusion:
	cdef:
		double complex [:] HyperDiffusionFactor
		double dissipation_order
		double truncation_order
		int truncation_number
		double complex [:] diffusion_factor

	cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
	cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, double dt)