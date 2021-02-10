import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

cpdef set_min_vapour(qp,qbar)
cpdef keSpectra(u,v)