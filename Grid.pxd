import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *

cdef class Grid:
    cdef:
        double nlats
        double nlons
        double n_layers
        int truncation_number
        double p1
        double p2
        double p3
        double p_ref
        double rsphere
        double omega
        double gravity
        double cp
        double Rd
        double Omega
        double kappa

        double [:] SphericalGrid
        double [:] longitude_list
        double [:] latitude_list
        double [:,:] lon
        double [:,:] lat
        double [:,:] longitude
        double [:,:] latitude
        double [:,:] Coriolis
