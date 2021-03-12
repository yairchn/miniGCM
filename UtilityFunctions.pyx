import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable
from libc.math cimport fmax, fmin, fabs, floor
import pylab as plt 

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

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void flux_constructor(Parameters Pr, Grid Gr, PrognosticVariable U,
                    PrognosticVariable V, PrognosticVariable Var):
    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t nx = Pr.nx
        Py_ssize_t ny = Pr.ny
        Py_ssize_t nl = Pr.n_layers
        Py_ssize_t ng = Gr.ng
        double phim3, phim2, phim1, phi, phip1, phip2, phip3, adv_vel

    with nogil:
        for i in xrange(ng,nx-ng-1):
            for j in xrange(ng,ny-ng-1):
                for k in xrange(nl):
                    phip3 = U.values[i+3,j,k]*Var.values[i+3,j,k]
                    phip2 = U.values[i+2,j,k]*Var.values[i+2,j,k]
                    phip1 = U.values[i+1,j,k]*Var.values[i+1,j,k]
                    phi   = U.values[i  ,j,k]*Var.values[i  ,j,k]
                    phim1 = U.values[i-1,j,k]*Var.values[i-1,j,k]
                    phim2 = U.values[i-2,j,k]*Var.values[i-2,j,k]
                    phim3 = U.values[i-3,j,k]*Var.values[i-3,j,k]

                    adv_vel = advection_velocity(U.values[i-1,j,k],
                                                 U.values[i,j,k],
                                                 U.values[i+1,j,k],
                                                 U.values[i+2,j,k])
                    # adv_vel = roe_velocity(U.values[i,j,k]*Var.values[i,j,k],
                    #                        U.values[i+1,j,k]*Var.values[i+1,j,k],
                    #                        Var.values[i,j,k],
                    #                        Var.values[i+1,j,k]
                    #                        )
                    if adv_vel>=0:
                        Var.ZonalFlux[i,j,k] = interp_weno5(phip2, phip1, phi, phim1, phim2)
                    else:
                        Var.ZonalFlux[i,j,k] = interp_weno5(phim1, phi, phip1, phip2, phip3)

                    phip3 = V.values[i,j+3,k]*Var.values[i,j+3,k]
                    phip2 = V.values[i,j+2,k]*Var.values[i,j+2,k]
                    phip1 = V.values[i,j+1,k]*Var.values[i,j+1,k]
                    phi   = V.values[i,j  ,k]*Var.values[i,j  ,k]
                    phim1 = V.values[i,j-1,k]*Var.values[i,j-1,k]
                    phim2 = V.values[i,j-2,k]*Var.values[i,j-2,k]
                    phim3 = V.values[i,j-3,k]*Var.values[i,j-3,k]
                    adv_vel = advection_velocity(V.values[i-1,j,k],
                                                 V.values[i,j,k],
                                                 V.values[i+1,j,k],
                                                 V.values[i+2,j,k])
                    # adv_vel = roe_velocity(V.values[i,j,k]*Var.values[i,j,k],
                    #                        V.values[i+1,j,k]*Var.values[i+1,j,k],
                    #                        Var.values[i,j,k],
                    #                        Var.values[i+1,j,k]
                    #                        )
                    if adv_vel>=0:
                        Var.MeridionalFlux[i,j,k] = interp_weno5(phip2, phip1, phi, phim1, phim2)
                    else:
                        Var.MeridionalFlux[i,j,k] = interp_weno5(phim1, phi, phip1, phip2, phip3)
    return

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void flux_constructor_fv(Parameters Pr, Grid Gr, PrognosticVariable U,
                    PrognosticVariable V, PrognosticVariable Var):
    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t nx = Pr.nx
        Py_ssize_t ny = Pr.ny
        Py_ssize_t nl = Pr.n_layers
        Py_ssize_t ng = Gr.ng

    with nogil:
        for i in xrange(ng,nx-ng-1):
            for j in xrange(ng,ny-ng-1):
                for k in xrange(nl):
                    if Var.name == 'u':
                        Var.ZonalFlux[i,j,k] = U.values[i,j,k]*Var.values[i,j,k]
                    else:
                        Var.ZonalFlux[i,j,k] = U.values[i,j,k]*(Var.values[i,j,k]+Var.values[i-1,j,k])

                    if Var.name == 'v':
                        Var.MeridionalFlux[i,j,k] = V.values[i,j,k]*Var.values[i,j,k]
                    else:
                        Var.MeridionalFlux[i,j,k] = V.values[i,j,k]*(Var.values[i,j,k]+Var.values[i,j-1,k])
    return

# @cython.wraparound(False)
# @cython.boundscheck(False)
# cdef void weno_flux_divergence(Parameters Pr, Grid Gr, PrognosticVariable U,
#                     PrognosticVariable V, PrognosticVariable Var):
#     cdef:
#         Py_ssize_t i, j, k
#         Py_ssize_t nx = Pr.nx
#         Py_ssize_t ny = Pr.ny
#         Py_ssize_t nl = Pr.n_layers
#         Py_ssize_t ng = Gr.ng
#         double roe_x_velocity_m, roe_x_velocity_p, roe_y_velocity_m, roe_y_velocity_p
#         double phim2, phim1, phi, phip1, phip2, weno_flux_m, weno_flux_p, fp, fm, adv_vel
#         double dxi = 1.0/Gr.dx
#         double dyi = 1.0/Gr.dy
#     with nogil:
#         for i in xrange(ng,nx-ng-1):
#             for j in xrange(ng,ny-ng-1):
#                 for k in xrange(nl):
#                     phip3 = U.values[i+3,j,k]*Var.values[i+3,j,k]
#                     phip2 = U.values[i+2,j,k]*Var.values[i+2,j,k]
#                     phip1 = U.values[i+1,j,k]*Var.values[i+1,j,k]
#                     phi   = U.values[i  ,j,k]*Var.values[i  ,j,k]
#                     phim1 = U.values[i-1,j,k]*Var.values[i-1,j,k]
#                     phim2 = U.values[i-2,j,k]*Var.values[i-2,j,k]
#                     phim3 = U.values[i-3,j,k]*Var.values[i-3,j,k]

#                     adv_vel = advection_velocity(U.values[i-1,j,k],
#                                                  U.values[i,j,k],
#                                                  U.values[i+1,j,k],
#                                                  U.values[i+2,j,k])
#                     interp_weno5(phim3, phim2, phim1, phi, phip1)

#                     Zonal_Flux
#                     Meridional_Flux
#                     if adv_vel>=0:
#                         Var.Weno_dFdx[i,j,k] = interp_weno5(phip2, phip1, phi, phim1, phim2)
#                     else:
#                         Var.Weno_dFdx[i,j,k] = interp_weno5(phim1, phi, phip1, phip2, phip3)

#                     adv_vel = advection_velocity(U.values[i+2,j,k] ,U.values[i+1,j,k] ,U.values[i  ,j,k] ,U.values[i-1,j,k])
#                     Var.Weno_dFdx[i,j,k] = (weno_flux_p - weno_flux_m)*dxi

#                     phip3 = V.values[i,j+3,k]*Var.values[i,j+3,k]
#                     phip2 = V.values[i,j+2,k]*Var.values[i,j+2,k]
#                     phip1 = V.values[i,j+1,k]*Var.values[i,j+1,k]
#                     phi   = V.values[i,j  ,k]*Var.values[i,j  ,k]
#                     phim1 = V.values[i,j-1,k]*Var.values[i,j-1,k]
#                     phim2 = V.values[i,j-2,k]*Var.values[i,j-2,k]
#                     phim3 = V.values[i,j-3,k]*Var.values[i,j-3,k]
#                     # roe_y_velocity_m = roe_velocity(V.values[i,j,k]*Var.values[i,j,k],
#                     #                                 V.values[i,j-1,k]*Var.values[i,j-1,k],
#                     #                                 Var.values[i,j,k],
#                     #                                 Var.values[i,j-1,k]
#                     #                                 )
#                     # roe_y_velocity_p = roe_velocity(V.values[i,j+1,k]*Var.values[i,j+1,k],
#                     #                                 V.values[i,j,k]*Var.values[i,j,k],
#                     #                                 Var.values[i,j+1,k],
#                     #                                 Var.values[i,j,k]
#                     #                                 )
#                     # if roe_x_velocity_m>=0:
#                     #     weno_flux_m = interp_weno5(phim3, phim2, phim1, phi, phip1)
#                     # else:
#                     #     weno_flux_m = interp_weno5(phip2, phip1, phi, phim1, phim2)

#                     # if roe_x_velocity_p>=0:
#                     #     weno_flux_p = interp_weno5(phim2, phim1, phi, phip1, phip2)
#                     # else:
#                     #     weno_flux_p = interp_weno5(phip3, phip2, phip1, phi, phim1)

#                     # Var.Weno_dFdy[i,j,k] = (weno_flux_p - weno_flux_m)*dyi

#                     adv_vel = advection_velocity(U.values[i+2,j,k] ,U.values[i+1,j,k] ,U.values[i  ,j,k] ,U.values[i-1,j,k])
#                     Var.Weno_dFdy[i,j,k] = (weno_flux_p - weno_flux_m)*dyi

#     return


# @cython.wraparound(False)
# @cython.boundscheck(False)
# cdef void weno_flux_divergence(Parameters Pr, Grid Gr, PrognosticVariable U,
#                     PrognosticVariable V, PrognosticVariable Var):
#     cdef:
#         Py_ssize_t i, j, k
#         Py_ssize_t nx = Pr.nx
#         Py_ssize_t ny = Pr.ny
#         Py_ssize_t nl = Pr.n_layers
#         Py_ssize_t ng = Gr.ng
#         double roe_x_velocity_m, roe_x_velocity_p, roe_y_velocity_m, roe_y_velocity_p
#         double phim2, phim1, phi, phip1, phip2, weno_flux_m, weno_flux_p, fp, fm, adv_vel
#         double dxi = 1.0/Gr.dx
#         double dyi = 1.0/Gr.dy
#     with nogil:
#         for i in xrange(ng,nx-ng-1):
#             for j in xrange(ng,ny-ng-1):
#                 for k in xrange(nl):
#                     phip3 = U.values[i+3,j,k]*Var.values[i+3,j,k]
#                     phip2 = U.values[i+2,j,k]*Var.values[i+2,j,k]
#                     phip1 = U.values[i+1,j,k]*Var.values[i+1,j,k]
#                     phi   = U.values[i  ,j,k]*Var.values[i  ,j,k]
#                     phim1 = U.values[i-1,j,k]*Var.values[i-1,j,k]
#                     phim2 = U.values[i-2,j,k]*Var.values[i-2,j,k]
#                     phim3 = U.values[i-3,j,k]*Var.values[i-3,j,k]

#                     # calcualte Roe velocity
#                     roe_x_velocity_m = roe_velocity(U.values[i,j,k]*Var.values[i,j,k],
#                                                     U.values[i-1,j,k]*Var.values[i-1,j,k],
#                                                     Var.values[i,j,k],
#                                                     Var.values[i-1,j,k]
#                                                     )
#                     roe_x_velocity_p = roe_velocity(U.values[i+1,j,k]*Var.values[i+1,j,k],
#                                                     U.values[i,j,k]*Var.values[i,j,k],
#                                                     Var.values[i+1,j,k],
#                                                     Var.values[i,j,k]
#                                                     )
#                     if roe_x_velocity_m>=0:
#                         weno_flux_m = interp_weno5(phim3, phim2, phim1, phi, phip1)
#                     else:
#                         weno_flux_m = interp_weno5(phip2, phip1, phi, phim1, phim2)

#                     if roe_x_velocity_p>=0:
#                         weno_flux_p = interp_weno5(phim2, phim1, phi, phip1, phip2)
#                     else:
#                         weno_flux_p = interp_weno5(phip3, phip2, phip1, phi, phim1)
#                     Var.Weno_dFdx[i,j,k] = (weno_flux_p - weno_flux_m)*dxi


#                     phip3 = V.values[i,j+3,k]*Var.values[i,j+3,k]
#                     phip2 = V.values[i,j+2,k]*Var.values[i,j+2,k]
#                     phip1 = V.values[i,j+1,k]*Var.values[i,j+1,k]
#                     phi   = V.values[i,j  ,k]*Var.values[i,j  ,k]
#                     phim1 = V.values[i,j-1,k]*Var.values[i,j-1,k]
#                     phim2 = V.values[i,j-2,k]*Var.values[i,j-2,k]
#                     phim3 = V.values[i,j-3,k]*Var.values[i,j-3,k]
#                     roe_y_velocity_m = roe_velocity(V.values[i,j,k]*Var.values[i,j,k],
#                                                     V.values[i,j-1,k]*Var.values[i,j-1,k],
#                                                     Var.values[i,j,k],
#                                                     Var.values[i,j-1,k]
#                                                     )
#                     roe_y_velocity_p = roe_velocity(V.values[i,j+1,k]*Var.values[i,j+1,k],
#                                                     V.values[i,j,k]*Var.values[i,j,k],
#                                                     Var.values[i,j+1,k],
#                                                     Var.values[i,j,k]
#                                                     )
#                     if roe_x_velocity_m>=0:
#                         weno_flux_m = interp_weno5(phim3, phim2, phim1, phi, phip1)
#                     else:
#                         weno_flux_m = interp_weno5(phip2, phip1, phi, phim1, phim2)

#                     if roe_x_velocity_p>=0:
#                         weno_flux_p = interp_weno5(phim2, phim1, phi, phip1, phip2)
#                     else:
#                         weno_flux_p = interp_weno5(phip3, phip2, phip1, phi, phim1)

#                     Var.Weno_dFdy[i,j,k] = (weno_flux_p - weno_flux_m)*dyi
#     return

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double interp_weno3(double phim1, double phi, double phip1) nogil:
    cdef:
        double p0,p1, beta0, beta1, alpha0, alpha1, alpha_sum_inv, w0, w1
    p0 = (-1.0/2.0) * phim1 + (3.0/2.0) * phi
    p1 = (1.0/2.0) * phi + (1.0/2.0) * phip1
    beta1 = (phip1 - phi) * (phip1 - phi)
    beta0 = (phi - phim1) * (phi - phim1)
    alpha0 = (1.0/3.0) /((beta0 + 1e-10) * (beta0 + 1.0e-10))
    alpha1 = (2.0/3.0)/((beta1 + 1e-10) * (beta1 + 1.0e-10))
    alpha_sum_inv = 1.0/(alpha0 + alpha1)
    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    return w0 * p0 + w1 * p1

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double interp_weno5(double phim2, double phim1, double phi, double phip1, double phip2) nogil:
    cdef:
        double p0, p1, p2, beta0, beta1, alpha0, alpha1, alpha_sum_inv, w0, w1
    p1 = (1.0/3.0) * phim2 - (7.0/6.0) * phim1 + (11.0/6.0) * phi
    p2 = -(1.0/6.0) * phim1 + (5.0/6.0) * phi + (1.0/3.0) * phip1
    p3 = (1.0/3.0) * phi + (5.0/6.0) * phip1 - (1.0/6.0) * phip2
    beta1 = ( (13.0/12.0)*(phim2 - 2.0*phim1 + phi)**2.0
             +(1.0/4.0)*(phim2 - 4.0*phim1 + 3.0*phi)**2.0 )
    beta2 = ( (13.0/12.0)*(phim1 - 2.0*phi + phip1)**2.0
             +(1.0/4.0)*(phim1 - phip1)**2.0)
    beta3 = ( (13.0/12.0)*(phi - 2.0*phip1 + phip2)**2.0
             +(1.0/4.0)*(3.0*phi - 4.0*phip1 + phip2)**2.0)

    alpha1 = (1.0/10.0)/((beta1 + 1e-10) * (beta1 + 1e-10))
    alpha2 = (3.0/ 5.0)/((beta2 + 1e-10) * (beta2 + 1e-10))
    alpha3 = (3.0/10.0)/((beta3 + 1e-10) * (beta3 + 1e-10))

    alpha_sum_inv = 1.0/(alpha1 + alpha2 + alpha3)
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv
    w3 = alpha3 * alpha_sum_inv
    return w1 * p1 + w2 * p2 + w3 * p3

cdef double roe_velocity(double fp, double fm, double varp, double varm) nogil:
    cdef:
        double roe_vel
    if fabs(varp-varm)>0.0:
        roe_vel = (fp-fm)/(varp-varm)
    else:
        roe_vel = 0.0
    return roe_vel

cdef double advection_velocity(double phim1, double phi, double phip1, double phip2) nogil:
    return (7.0/12.0)*(phi + phip1 ) -(1.0/12.0)*(phim1 + phip2)


cpdef axisymmetric_mean(xc, yc, data):
    cdef:
        Py_ssize_t r
        double [:] axi_data

    y = np.arange(np.shape(data)[0])
    x = np.arange(np.shape(data)[1])
    X, Y = np.meshgrid(x,y)
    nr = (np.min([xc,np.size(data,0)-xc,yc, np.size(data,1) - yc])-1).astype(np.int)
    axi_data = np.zeros(nr+1)
    axi_data[0] = data[xc, yc]
    for r in range(nr):
        mask = np.logical_and((X-xc)**2 + (Y-yc)**2 > r**2,
                              (X-xc)**2 + (Y-yc)**2 <= (r+1)**2)
        axi_data[r+1] = np.nanmean(data[mask])

    return axi_data
