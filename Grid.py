
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *

class Grid:
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
        self.Lv = namelist['thermodynamics']['latent_heat_vap']
        # self.R = namelist['planet']['planet_radius']
        self.Omega = namelist['planet']['omega_rotation']
        
        self.kappa = self.Rd/self.cp
        self.p21 = self.p2 - self.p1
        self.p12 = self.p2 + self.p1
        self.p32 = self.p3 - self.p2
        self.p23 = self.p3 + self.p2
        

		# setup up spherical harmonic instance, set lats/lons of grid
        self.SphericalGrid = sph.Spharmt(self.nlons, self.nlats, self.truncation_number, self.rsphere, gridtype='gaussian')
		# build the physical space grid
        self.lon, self.lat = np.meshgrid(self.SphericalGrid.lons, self.SphericalGrid.lats) # these are in radians
        self.longitude = np.degrees(self.lon)
        self.latitude  = np.degrees(self.lat)
        self.longitude_list = self.longitude[1,:]
        self.latitude_list = self.latitude[:,1]
        # Yair - make this a 2D matrix and convert to spectral ? 
        self.Coriolis = 2.0*self.Omega*np.sin(self.lat)
        
		#print("lats: ",lats*360./(2.*pi))
		# temperature perturbation
        lats_ref   = np.radians(0)
        lats_ref   = np.radians(50)
        lats_ref   = np.radians(40)
        lons_ref   = np.radians(180)
        delta_lats = np.radians(10)
        delta_lons = np.radians(30)
        delta_lons = np.radians(10)
        delta_lats = np.radians(30)
        delta_lons = np.radians(30)

        return

