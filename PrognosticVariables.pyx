#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import cython
from concurrent.futures import ThreadPoolExecutor
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
import Microphysics
cimport Microphysics
from NetCDFIO cimport NetCDFIO_Stats
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from UtilityFunctions import interp_weno3, roe_velocity

cdef extern from "tendency_functions.h":
    void rhs_qt(double* p, double* qt, double* u, double* v, double* wp,
                double* qt_mp, double* qt_sur, double* rhs_qt, double* u_qt, double* v_qt,
                Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax, Py_ssize_t k) nogil

    void rhs_T(double cp, double* p, double* gz, double* T, double* u, double* v, double* wp,
               double* T_mp, double* T_sur, double* T_forc, double* rhs_T, double* u_T,
               double* v_T, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax, Py_ssize_t k) nogil

    void vertical_uv_fluxes(double* p, double* gz, double* vort, double* f,
                            double* u, double* v, double* wp, double* ke, double* wdudp_up, double* wdvdp_up,
                            double* wdudp_dn, double* wdvdp_dn, double* e_dry, double* u_vort, double* v_vort,
                            ssize_t imax, ssize_t jmax, ssize_t kmax, ssize_t k) nogil



cdef class PrognosticVariable:
    def __init__(self, nx, ny, nl, kind, name, units):
        self.kind = kind
        self.name = name
        self.units = units

        self.values   = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.old      = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.now      = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.SurfaceFlux = np.zeros((nx,ny)   ,dtype=np.float64, order='c')
        self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.float64, order='c')
        if name=='T' or name=='u' or name=='v':
            self.forcing = np.zeros((nx,ny,nl),dtype = np.float64, order='c')
        if name=='T' or name=='QT':
            self.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')


        return

cdef class PrognosticVariables:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.U  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   'zonal velocity'      ,'u'  ,'1/s')
        self.V  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   'meridionla velocity' ,'v'  ,'1/s')
        self.T  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   'Temperature'         ,'T'  ,'K')
        self.QT = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   'Specific Humidity'   ,'QT' ,'K')
        self.P  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, 'Pressure'            ,'p'  ,'pasc')
        return

    cpdef initialize(self, Parameters Pr):
        self.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_T')
        Stats.add_global_mean('global_mean_QT')
        Stats.add_zonal_mean('zonal_mean_P')
        Stats.add_zonal_mean('zonal_mean_T')
        Stats.add_zonal_mean('zonal_mean_QT')
        Stats.add_zonal_mean('zonal_mean_U')
        Stats.add_zonal_mean('zonal_mean_V')
        Stats.add_meridional_mean('meridional_mean_U')
        Stats.add_meridional_mean('meridional_mean_V')
        Stats.add_meridional_mean('meridional_mean_P')
        Stats.add_meridional_mean('meridional_mean_T')
        Stats.add_meridional_mean('meridional_mean_QT')
        return

    cpdef set_bcs(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nx
            Py_ssize_t ny = Pr.ny
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t ng = Gr.ng
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        for i in xrange(1,Gr.gw):
            for j in xrange(1,Gr.gw):
                self.values[i,start_high] = 0.0
                self.values[i,start_low] = 0.0
                for k in xrange(nl):
                    self.values[i,start_high + k +1] = self.values[i,start_high  - k]
                    self.values[i,start_low - k] = self.values[i,start_low + 1 + k]
        return
    # quick utility to set arrays with values in the "new" arrays
    cpdef set_old_with_now(self):
        self.U.old  = self.U.now.copy()
        self.V.old  = self.V.now.copy()
        self.T.old  = self.T.now.copy()
        self.QT.old = self.QT.now.copy()
        self.P.old  = self.P.now.copy()
        return

    cpdef set_now_with_tendencies(self):
        self.U.now  = self.U.tendency.copy()
        self.V.now  = self.V.tendency.copy()
        self.T.now  = self.T.tendency.copy()
        self.QT.now = self.QT.tendency.copy()
        self.P.now  = self.P.tendency.copy()
        return

    cpdef reset_pressures_and_bcs(self, Parameters Pr, DiagnosticVariables DV):
        cdef:
            Py_ssize_t nl = Pr.n_layers

        DV.gZ.values.base[:,:,nl] = np.zeros_like(self.P.values.base[:,:,nl])
        DV.Wp.values.base[:,:,0]  = np.zeros_like(self.P.values.base[:,:,nl])
        for k in range(nl):
            self.P.values.base[:,:,k] = np.add(np.zeros_like(self.P.values.base[:,:,k]),self.P_init[k])
            self.P.spectral.base[:,k] = np.add(np.zeros_like(self.P.spectral.base[:,k]),self.P_init[k])
        return
    # this should be done in time intervals and save each time new files,not part of stats

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_T', self.T.values)
        Stats.write_global_mean('global_mean_QT', self.QT.values)
        Stats.write_zonal_mean('zonal_mean_P',self.P.values[:,:,1:4])
        Stats.write_zonal_mean('zonal_mean_T',self.T.values)
        Stats.write_zonal_mean('zonal_mean_QT',self.QT.values)
        Stats.write_zonal_mean('zonal_mean_U',self.U.values)
        Stats.write_zonal_mean('zonal_mean_V',self.V.values)
        Stats.write_meridional_mean('meridional_mean_P',self.P.values[:,:,1:4])
        Stats.write_meridional_mean('meridional_mean_T',self.T.values)
        Stats.write_meridional_mean('meridional_mean_QT',self.QT.values)
        Stats.write_meridional_mean('meridional_mean_U',self.U.values)
        Stats.write_meridional_mean('meridional_mean_V',self.V.values)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t nl = Pr.n_layers
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'U',                 self.U.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'V',                 self.V.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Temperature',       self.T.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'dQTdt',             self.QT.mp_tendency[:,:,0:nl])
        Stats.write_2D_variable(Pr, int(TS.t),    'Pressure',          self.P.values[:,:,nl])
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nx
            Py_ssize_t ny = Pr.ny
            Py_ssize_t nl = Pr.n_layers
            double fu, fv, duu_dx, duv_dy, dvu_dx, dvv_dy, duT_dx, dvT_dy
            double duQT_dx, dvQT_dy, dgZ_dx, dgZ_dy, Wp_dgZ_dp, T_sur_flux, QT_sur_flux,
            double U_sur_flux, V_sur_flux
            double dxi = 1.0/Gr.dx
            double dyi = 1.0/Gr.dy

            # double [:,:] duu_dx = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] duv_dy = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] dvu_dx = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] dvv_dy = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] duT_dx = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] dvT_dy = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] duQT_dx = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] dvQT_dy = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] fv = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] fu = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] dgZ_dx = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] dgZ_dy = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] Wp_dgZ_dp = np.zeros((nx, ny), dtype = np.float64, order ='c')
            # double [:,:] T_sur_flux  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            # double [:,:] QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
            # double [:,:] U_sur_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
            # double [:,:] V_sur_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')


        # duu_dx = weno3_flux_divergence(PV.U.values,PV.U.values, dxi, nx, ny, nl)
        # duv_dy = weno3_flux_divergence(PV.U.values,PV.V.values, dyi, nx, ny, nl)
        # dvu_dx = weno3_flux_divergence(PV.V.values,PV.U.values, dxi, nx, ny, nl)
        # dvv_dy = weno3_flux_divergence(PV.V.values,PV.V.values, dyi, nx, ny, nl)
        # duT_dx = weno3_flux_divergence(PV.U.values,PV.T.values, dxi, nx, ny, nl)
        # dvT_dy = weno3_flux_divergence(PV.V.values,PV.T.values, dyi, nx, ny, nl)
        # duQT_dx = weno3_flux_divergence(PV.U.values,PV.QT.values, dxi, nx, ny, nl)
        # dvQT_dy = weno3_flux_divergence(PV.V.values,PV.QT.values, dyi, nx, ny, nl)
        with nogil:
            for i in range(Pr.nx):
                for j in range(Pr.ny):
                    PV.P.tendency[i,j,nl] = (DV.Wp.values[i,j,nl-1]
                               + (PV.P.values[i,j,nl]-PV.P.values[i,j,nl-1])*DV.Divergence.values[i,j,nl-1])
                    for k in range(nl):
                        if k==nl-1:
                            U_sur_flux  = PV.U.SurfaceFlux[i,j]
                            V_sur_flux  = PV.V.SurfaceFlux[i,j]
                            T_sur_flux  = PV.T.SurfaceFlux[i,j]
                            QT_sur_flux = PV.QT.SurfaceFlux[i,j]

                        duu_dx = 0.5*(PV.U.values[i+1,j,k]*PV.U.values[i+1,j,k]-PV.U.values[i-1,j,k]*PV.U.values[i-1,j,k])*dxi
                        duv_dy = 0.5*(PV.U.values[i,j+1,k]*PV.U.values[i,j+1,k]-PV.U.values[i,j-1,k]*PV.U.values[i,j-1,k])*dyi
                        dvu_dx = 0.5*(PV.U.values[i+1,j,k]*PV.V.values[i+1,j,k]-PV.U.values[i-1,j,k]*PV.V.values[i-1,j,k])*dxi
                        dvv_dy = 0.5*(PV.V.values[i,j+1,k]*PV.V.values[i,j+1,k]-PV.V.values[i,j-1,k]*PV.V.values[i,j-1,k])*dyi
                        duT_dx = 0.5*(PV.U.values[i+1,j,k]*PV.T.values[i+1,j,k]-PV.U.values[i-1,j,k]*PV.T.values[i-1,j,k])*dxi
                        dvT_dy = 0.5*(PV.V.values[i,j+1,k]*PV.T.values[i,j+1,k]-PV.V.values[i,j-1,k]*PV.T.values[i,j-1,k])*dyi
                        duQT_dx = 0.5*(PV.U.values[i+1,j,k]*PV.QT.values[i+1,j,k]-PV.U.values[i-1,j,k]*PV.QT.values[i-1,j,k])*dxi
                        dvQT_dy = 0.5*(PV.V.values[i,j+1,k]*PV.QT.values[i,j+1,k]-PV.V.values[i,j-1,k]*PV.QT.values[i,j-1,k])*dyi
                        # dgZ_dx = (DV.gZ.values[i+1,j,k]-DV.gZ.values[i-1,j,k])*dxi
                        # dgZ_dy = (DV.gZ.values[i,j+1,k]-DV.gZ.values[i,j-1,k])*dyi
                        # Wp_dgZ_dp = (DV.Wp.values[i,j,k+1]+DV.Wp.values[i,j,k])/2.0*(DV.gZ.values[i,j,k+1] - DV.gZ.values[i,j,k])/(PV.P.values[i,j,k+1] - PV.P.values[i,j,k])
                        fv = Pr.Coriolis*PV.V.values[i,j,k]
                        fu = Pr.Coriolis*PV.U.values[i,j,k]
                        dgZ_dx = (DV.gZ.values[i+1,j,k]-DV.gZ.values[i-1,j,k])*dxi
                        dgZ_dy = (DV.gZ.values[i,j+1,k]-DV.gZ.values[i,j-1,k])*dyi
                        Wp_dgZ_dp = ((DV.Wp.values[i,j,k+1]+DV.Wp.values[i,j,k])/2.0*
                            (DV.gZ.values[i,j,k+1]-DV.gZ.values[i,j,k-1])
                           /(PV.P.values[i,j,k+1]-PV.P.values[i,j,k-1]))

                        PV.U.tendency[i,j,k]  = - duu_dx - duv_dy - fv -dgZ_dx
                        PV.V.tendency[i,j,k]  = - dvu_dx - dvv_dy + fu -dgZ_dy

                        PV.T.tendency[i,j,k]  = - duT_dx - dvT_dy - Wp_dgZ_dp + PV.T.mp_tendency[i,j,k] + PV.T.forcing[i,j,k] + T_sur_flux

                        if Pr.moist_index > 0.0:
                            PV.QT.tendency[i,j,k]  = - duQT_dx - dvQT_dy + PV.QT.mp_tendency[i,j,k] + QT_sur_flux
        return

    # cpdef weno3_flux_divergence(U, Var, dxi, nx, ny, nl):
    #     cdef:
    #         double roe_x_velocity_m, roe_x_velocity_p, roe_y_velocity_m, roe_y_velocity_p
    #         double phim2, phim1, phi, phip1, phip2, weno_flux_m, weno_flux_p
    #         double [:,:,:] weno_fluxdivergence_x
    #         double [:,:,:] weno_fluxdivergence_y

    #     with nogil:
    #         for i in range(nx):
    #             for j in range(ny):
    #                 for k in range(nz):
    #                     # calcualte Roe velocity
    #                     roe_x_velocity_m = roe_velocity(U[i,j,k]*Var[i,j,k],U[i-1,j,k]*Var[i-1,j,k],
    #                                                     Var[i,j,k],Var[i-1,j,k])
    #                     roe_x_velocity_p = roe_velocity(U[i+1,j,k]*Var[i+1,j,k],U[i,j,k]*Var[i,j,k],
    #                                                     Var[i+1,j,k],Var[i,j,k])
    #                     phip2 = U[i+2,j,k]*Var[i+2,j,k]
    #                     phip1 = U[i+1,j,k]*Var[i+1,j,k]
    #                     phi   = U[i  ,j,k]*Var[i  ,j,k]
    #                     phim1 = U[i-1,j,k]*Var[i-1,j,k]
    #                     phim2 = U[i-2,j,k]*Var[i-2,j,k]

    #                     if roe_x_velocity_m>=0:
    #                         weno_flux_m = interp_weno3(phim2, phim1, phi)
    #                     else:
    #                         weno_flux_m = interp_weno3(phip1, phi, phim1)

    #                     if roe_x_velocity_p>=0:
    #                         weno_flux_p = interp_weno3(phim1, phi, phip1)
    #                     else:
    #                         weno_flux_p = interp_weno3(phip2, phip1, phi)

    #                     weno_fluxdivergence_x[i,j,k] = (weno_flux_p - weno_flux_m)*dxi

    #                     roe_y_velocity_m = roe_velocity(V[i,j,k]*Var[i,j,k],V[i,j-1,k]*Var[i,j-1,k],
    #                                                     Var[i,j,k],Var[i,j-1,k])
    #                     roe_y_velocity_p = roe_velocity(V[i,j+1,k]*Var[i,j+1,k],V[i,j,k]*Var[i,j,k],
    #                                                     Var[i,j+1,k],Var[i,j,k])
    #                     phip2 = V[i,j+2,k]*Var[i,j+2,k]
    #                     phip1 = V[i,j+1,k]*Var[i,j+1,k]
    #                     phi   = V[i,j  ,k]*Var[i,j  ,k]
    #                     phim1 = V[i,j-1,k]*Var[i,j-1,k]
    #                     phim2 = V[i,j-2,k]*Var[i,j-2,k]

    #                     if roe_y_velocity_m>=0:
    #                         weno_flux_m = interp_weno3(phim2, phim1, phi)
    #                     else:
    #                         weno_flux_m = interp_weno3(phip1, phi, phim1)

    #                     if roe_y_velocity_p>=0:
    #                         weno_flux_p = interp_weno3(phim1, phi, phip1)
    #                     else:
    #                         weno_flux_p = interp_weno3(phip2, phip1, phi)

    #                     weno_fluxdivergence_y[i,j,k] = (weno_flux_p - weno_flux_m)*dyi

    #     return weno_fluxdivergence_x, weno_fluxdivergence_y

