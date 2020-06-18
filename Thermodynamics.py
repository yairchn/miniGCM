
import time
import scipy as sc
import numpy as np
from math import *

class Thermodynamics:
	def __init__(self, namelist):
		self.cp = namelist['thermodynamics']['heat_capacity']
		self.Rd = namelist['thermodynamics']['ideal_gas_constant']
		self.kappa = self.R/self.cp
		return

	def initialize(self):
		return

	def update(self):
		return