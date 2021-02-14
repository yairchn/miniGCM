import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
import shtns
import sphTrans as sph
from math import *
import os

cdef class Parameters:
    cdef:
        Py_ssize_t nlats
        Py_ssize_t nlons
        Py_ssize_t n_layers
        int truncation_order
        int truncation_number
        double moist_index
        double dissipation_order
        double rho1
        double rho2
        double rho3
        double rsphere
        double omega
        double g
        double Omega
        double k_a
        double k_s
        double k_f
        double k_b
        double DH_y
        double H_equator
        double Dtheta_z
        double Hbar0
        double P_hw
        double phi_hw
        double H_pole
        double init_k
        double max_ss
        double rho_w
        double Cd
        double Ch
        double Cq
        double dH_s
        double H_min
        double dphi_s
        double efold
        double inoise
        double noise_amp
        double [:] rho

        str case
        str surface_model
        str uuid
        str casename
        str outpath
        str logfilename
        str thermodynamics_type