import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from PrognosticVariables cimport PrognosticVariable
from libc.math cimport fmax, fmin, fabs, floor

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

cdef weno3_flux_divergence(Parameters Pr, Grid Gr, double *U, double *V, double *Var):
        cdef:
            int i, j, k
            int nx = Pr.nx
            int ny = Pr.ny
            int nl = Pr.n_layers
            int ng = Gr.ng
            double roe_x_velocity_m, roe_x_velocity_p, roe_y_velocity_m, roe_y_velocity_p
            double phim2, phim1, phi, phip1, phip2, weno_flux_m, weno_flux_p, fp, fm
            double dxi = 1.0/Gr.dx
            double dyi = 1.0/Gr.dy
            double [:,:,:] weno_fluxdivergence_x = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            double [:,:,:] weno_fluxdivergence_y = np.zeros((nx,ny,nl),dtype=np.float64, order='c')

        with nogil:
            for i in xrange(ng,nx-ng-1):
                for j in xrange(ng,ny-ng-1):
                    for k in xrange(nl):
                        # calcualte Roe velocity
                        # roe_x_velocity_m = roe_velocity(U[i,j,k]*Var[i,j,k],U[i-1,j,k]*Var[i-1,j,k], Var[i,j,k],Var[i-1,j,k])
                        # roe_x_velocity_p = roe_velocity(U[i+1,j,k]*Var[i+1,j,k],U[i,j,k]*Var[i,j,k],
                        #                                 Var[i+1,j,k],Var[i,j,k])
                        # phip2 = U[i+2,j,k]*Var[i+2,j,k]
                        # phip1 = U[i+1,j,k]*Var[i+1,j,k]
                        # phi   = U[i  ,j,k]*Var[i  ,j,k]
                        # phim1 = U[i-1,j,k]*Var[i-1,j,k]
                        # phim2 = U[i-2,j,k]*Var[i-2,j,k]

                        # if roe_x_velocity_m>=0:
                        #     weno_flux_m = interp_weno3(phim2, phim1, phi)
                        # else:
                        #     weno_flux_m = interp_weno3(phip1, phi, phim1)

                        # if roe_x_velocity_p>=0:
                        #     weno_flux_p = interp_weno3(phim1, phi, phip1)
                        # else:
                        #     weno_flux_p = interp_weno3(phip2, phip1, phi)

                        # weno_fluxdivergence_x[i,j,k] = (weno_flux_p - weno_flux_m)*dxi

                        # roe_y_velocity_m = roe_velocity(V[i,j,k]*Var[i,j,k],V[i,j-1,k]*Var[i,j-1,k],
                        #                                 Var[i,j,k],Var[i,j-1,k])
                        # roe_y_velocity_p = roe_velocity(V[i,j+1,k]*Var[i,j+1,k],V[i,j,k]*Var[i,j,k],
                        #                                 Var[i,j+1,k],Var[i,j,k])
                        # phip2 = V[i,j+2,k]*Var[i,j+2,k]
                        # phip1 = V[i,j+1,k]*Var[i,j+1,k]
                        # phi   = V[i,j  ,k]*Var[i,j  ,k]
                        # phim1 = V[i,j-1,k]*Var[i,j-1,k]
                        # phim2 = V[i,j-2,k]*Var[i,j-2,k]

                        # if roe_y_velocity_m>=0:
                        #     weno_flux_m = interp_weno3(phim2, phim1, phi)
                        # else:
                        #     weno_flux_m = interp_weno3(phip1, phi, phim1)

                        # if roe_y_velocity_p>=0:
                        #     weno_flux_p = interp_weno3(phim1, phi, phip1)
                        # else:
                        #     weno_flux_p = interp_weno3(phip2, phip1, phi)

                        # weno_fluxdivergence_y[i,j,k] = (weno_flux_p - weno_flux_m)*dyi

        return weno_fluxdivergence_x, weno_fluxdivergence_y

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

cdef double roe_velocity(double fp, double fm, double varp, double varm) nogil:
    cdef:
        double roe_vel
    if fabs(varp-varm)>0.0:
        roe_vel = (fp-fm)/(varp-varm)
    else:
        roe_vel = 0.0
    return roe_vel