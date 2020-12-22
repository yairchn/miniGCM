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
    cdef:
        Py_ssize_t nlats
        Py_ssize_t nlons
        Py_ssize_t n_layers
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

        object Spharmt
        double [:] longitude_list
        double [:] latitude_list
        double [:,:] lon
        double [:,:] lat
        double [:,:] longitude
        double [:,:] latitude
        double [:,:] Coriolis
