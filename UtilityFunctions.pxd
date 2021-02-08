import cython
from concurrent.futures import ThreadPoolExecutor
import os
from Grid cimport Grid
import numpy as np
cimport numpy as np
import scipy as sc
from math import *
import sys
from Parameters cimport Parameters

cpdef set_min_vapour(qp,qbar)
cpdef keSpectra(u,v)