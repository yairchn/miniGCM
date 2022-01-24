import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
import shtns
import sphTrans as sph
from Parameters cimport Parameters

cdef class Grid:
	cdef:
		object SphericalGrid
		double [:] longitude_list
		double [:] latitude_list
		double [:] lat_weights
		double [:,:] lon
		double [:,:] lat
		double [:,:] longitude
		double [:,:] latitude
		double [:,:] Coriolis
		double [:,:] dx
		double [:,:] dy
		double complex [:] laplacian