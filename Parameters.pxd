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
        double H1
        double H2
        double H3
        double QT1
        double QT2
        double QT3
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
        double dT_s
        double T_min
        double Rd
        double Rv
        double Lv
        double qv_star0
        double T_0
        double eps_v

        str noise_type
        str case
        str surface_model
        str uuid
        str casename
        str outpath
        str logfilename
        str thermodynamics_type