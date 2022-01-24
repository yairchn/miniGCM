import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
import shtns
import sphTrans as sph
from Parameters cimport Parameters
from math import *

cdef class Grid:
	def __init__(self, Parameters Pr, namelist):
		self.SphericalGrid = sph.Spharmt(Pr.nlons, Pr.nlats, Pr.truncation_number, Pr.rsphere, gridtype='gaussian')
		self.lon, self.lat = np.meshgrid(self.SphericalGrid.lons, self.SphericalGrid.lats) # these are in radians
		self.longitude = np.degrees(self.lon)
		self.latitude  = np.degrees(self.lat)
		self.longitude_list = self.longitude[1,:]
		self.latitude_list = self.latitude[:,1]
		self.Coriolis = 2.0*Pr.Omega*np.sin(self.lat)
		self.dx = np.abs(np.multiply(np.multiply(np.gradient(self.lon,axis=1),Pr.rsphere),np.cos(self.lat)))
		self.dy = np.abs(np.multiply(np.gradient(self.lat,axis=0),Pr.rsphere))
		self.laplacian = self.SphericalGrid.lap
		self.lat_weights = np.cos(2.*pi*np.linspace(-90.,90.,Pr.nlats)/360.)
		self.lat_weights /= np.sum(self.lat_weights)
		return

