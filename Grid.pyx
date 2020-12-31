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
    def __init__(self, Parameters Pr, namelist):
        self.SphericalGrid = sph.Spharmt(Pr.nlons, Pr.nlats, Pr.truncation_number, Pr.rsphere, gridtype='gaussian')
        self.lon, self.lat = np.meshgrid(self.SphericalGrid.lons, self.SphericalGrid.lats) # these are in radians
        self.longitude = np.degrees(self.lon)
        self.latitude  = np.degrees(self.lat)
        self.longitude_list = self.longitude[1,:]
        self.latitude_list = self.latitude[:,1]
        self.Coriolis = 2.0*Pr.Omega*np.sin(self.lat)
        return

