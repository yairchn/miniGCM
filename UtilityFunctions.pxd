import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

cpdef set_min_vapour(qp,qbar)
cpdef keSpectra(Grid Gr, u, v)
cpdef inv_operator_vt(Grid Gr, Parameters Pr, miu, Omega, epsilon, zeta_bar_m, Der2_dir, cos_lat_sqr)