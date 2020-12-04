
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *

class Grid:
    def __init__(self, Pr, namelist):

        self.p21 = Pr.p2 - Pr.p1
        self.p12 = Pr.p2 + Pr.p1
        self.p32 = Pr.p3 - Pr.p2
        self.p23 = Pr.p3 + Pr.p2

		# setup up spherical harmonic instance, set lats/lons of grid
        self.SphericalGrid = sph.Spharmt(Pr.nlons, Pr.nlats, Pr.truncation_number, Pr.rsphere, gridtype='gaussian')
		# build the physical space grid
        self.lon, self.lat = np.meshgrid(self.SphericalGrid.lons, self.SphericalGrid.lats) # these are in radians
        self.longitude = np.degrees(self.lon)
        self.latitude  = np.degrees(self.lat)
        self.longitude_list = self.longitude[1,:]
        self.latitude_list = self.latitude[:,1]
        self.Coriolis = 2.0*Pr.Omega*np.sin(self.lat)
        return

