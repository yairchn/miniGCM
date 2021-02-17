import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

# make sure that total moisture content is non-negative
cpdef set_min_vapour(qp,qbar):
    qtot = qp + qbar
    qtot[qtot<0] = 0
    return (qtot-qbar)

# Function for plotting KE spectra
cpdef keSpectra(Grid Gr, u, v):
    uk = Gr.grdtospec(u)
    vk = Gr.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(Gr.l)+1)
    k = np.arange(np.amax(Gr.l)+1)
    for i in range(0,np.amax(Gr.l)):
        Ek[i] = np.sum(Esp[np.logical_and(Gr.l>=i-0.5 , Gr.l<i+0.5)])
    return [Ek,k]

# cdef double interp_weno3(double phim1, double phi, double phip1) nogil:
#     cdef:
#         double p0,p1, beta0, beta1, alpha0, alpha1, alpha_sum_inv, w0, w1
#     p0 = (-1.0/2.0) * phim1 + (3.0/2.0) * phi
#     p1 = (1.0/2.0) * phi + (1.0/2.0) * phip1
#     beta1 = (phip1 - phi) * (phip1 - phi)
#     beta0 = (phi - phim1) * (phi - phim1)
#     alpha0 = (1.0/3.0) /((beta0 + 1e-10) * (beta0 + 1.0e-10))
#     alpha1 = (2.0/3.0)/((beta1 + 1e-10) * (beta1 + 1.0e-10))
#     alpha_sum_inv = 1.0/(alpha0 + alpha1)
#     w0 = alpha0 * alpha_sum_inv
#     w1 = alpha1 * alpha_sum_inv
#     return w0 * p0 + w1 * p1

# cdef double roe_velocity(double fp, double fm, double varp, double varm) nogil:
#     cdef:
#         double roe_vel
#     if fabs(varp-varm)>0.0:
#         roe_vel = (fp-fm)/(varp-varm)
#     else:
#         roe_vel = 0.0
#     return roe_vel