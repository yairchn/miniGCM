import cython
import matplotlib.pyplot as plt
from math import *
import numpy as np
import scipy as sc
import shtns
import sphTrans as sph
# cimport sphTrans as sph
# from sphTrans import Spharmt
# from sphTrans cimport Spharmt
import time
import sys

cdef class Grid:
    def __init__(self, namelist):
        self.nlats             = namelist['grid']['number_of_latitute_points']
        self.nlons             = namelist['grid']['number_of_longitude_points']
        self.n_layers          = namelist['grid']['number_of_layers']
        self.truncation_number = int(self.nlons/3)
        self.p1                = namelist['grid']['p1']
        self.p2                = namelist['grid']['p2']
        self.p3                = namelist['grid']['p3']
        self.p_ref             = namelist['grid']['p_ref']
        self.rsphere           = namelist['planet']['planet_radius']
        self.omega             = namelist['planet']['omega_rotation']
        self.gravity           = namelist['planet']['gravity']
        self.cp = namelist['thermodynamics']['heat_capacity']
        self.Rd = namelist['thermodynamics']['ideal_gas_constant']
        self.Omega = namelist['planet']['omega_rotation']
        self.kappa = self.Rd/self.cp
		# setup up spherical harmonic instance, set lats/lons of grid
        print(sph.Spharmt(self.nlons, self.nlats, self.truncation_number, self.rsphere, gridtype='gaussian').lons)
        # self.Spharmt = sph.Spharmt(self.nlons, self.nlats, self.truncation_number, self.rsphere, gridtype='gaussian')
		# build the physical space grid
        self.lon, self.lat = np.meshgrid(self.Spharmt.lons, self.Spharmt.lats) # these are in radians
        self.longitude = np.degrees(self.lon)
        self.latitude  = np.degrees(self.lat)
        self.longitude_list = self.longitude[1,:]
        self.latitude_list = self.latitude[:,1]
        self.Coriolis = 2.0*self.Omega*np.sin(self.lat)

        return

