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
        double p1
        double p2
        double p3
        double p4
        double p5
        double T1
        double T2
        double T3
        double T4
        double T5
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
        double QT_0
        double eps_v
        double kappa
        double sigma_b
        double k_a
        double k_s
        double k_f
        double k_b
        double DT_y
        double T_equator
        double Dtheta_z
        double Tbar0
        double P_hw
        double phi_hw
        double T_pole
        double init_k
        double max_ss
        double rho_w
        double Cd
        double Ch
        double Cq
        double dT_s
        double T_min
        double dphi_s
        double efold_meso
        double efold_grid
        double inoise
        double noise_amp

        str case
        str noise_type
        str surface_model
        str uuid
        str casename
        str outpath
        str logfilename
        str thermodynamics_type
