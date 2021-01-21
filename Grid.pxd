import cython
import matplotlib.pyplot as plt
from math import *
import numpy as np
import scipy as sc
import shtns
import sphTrans as sph
import time
import sys
from Parameters cimport Parameters

cdef class Grid:
	cdef:
		object SphericalGrid
		double [:] longitude_list
		double [:] latitude_list
		double [:,:] lon
		double [:,:] lat
		double [:,:] longitude
		double [:,:] latitude
		double [:,:] Coriolis
		double [:,:] dx
		double [:,:] dy