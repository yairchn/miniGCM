import cython
import matplotlib.pyplot as plt
from math import *
import numpy as np
import scipy as sc
import shtns
import sphTrans as sph
import time
import sys

cdef class Grid:
    def __init__(self, namelist):
		# setup up spherical harmonic instance, set lats/lons of grid
        self.SphericalGrid = sph.Spharmt(self.nlons, self.nlats, self.truncation_number, self.rsphere, gridtype='gaussian')
        self.lon, self.lat = np.meshgrid(self.SphericalGrid.lons, self.SphericalGrid.lats) # these are in radians
        self.longitude = np.degrees(self.lon)
        self.latitude  = np.degrees(self.lat)
        self.longitude_list = self.longitude[1,:]
        self.latitude_list = self.latitude[:,1]
        self.Coriolis = 2.0*self.Omega*np.sin(self.lat)
        self.Coriolis = 2.0*Pr.Omega*np.sin(self.lat)
        return

