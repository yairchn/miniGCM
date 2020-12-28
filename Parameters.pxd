import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *

cdef class Parameters:
    cdef:
        int nlats
        int nlons
        int n_layers
        int truncation_number
        double p1
        double p2
        double p3
        double p_ref
        double rsphere
        double omega
        double g
        double Omega
        double cp
        double Rd
        double Rv
        double Lv
        double qv_star0
        double T_0
        double eps_v
        double kappa
        str case
        str surface_model