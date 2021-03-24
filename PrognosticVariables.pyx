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
from UtilityFunctions cimport *

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
        self.HyperDiffusion = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.ZonalFlux = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.MeridionalFlux = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.SurfaceFlux = np.zeros((nx,ny)   ,dtype=np.float64, order='c')
        self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.float64, order='c')
        self.forcing = np.zeros((nx,ny,nl),dtype = np.float64, order='c')
        # if name=='T' or name=='u' or name=='v':
        if name=='T' or name=='QT':
            self.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')


    cpdef set_bcs(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t i, j, k, q, p
            Py_ssize_t nx = Pr.nx
            Py_ssize_t ny = Pr.ny
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t ng = Gr.ng

        if self.name == 'u':
            for i in range(ng):
                # full
                self.values.base[i,:,:]         = self.values.base[nx+i,:,:]
                self.values.base[ng+nx+1+i,:,:] = self.values.base[ng+i+1,:,:]
                # half
                self.values.base[:,i,:]       = self.values.base[:,ny+i,:]
                self.values.base[:,ng+ny+i,:] = self.values.base[:,ng+i,:]

        elif self.name == 'v':
            for i in range(ng):
                # full
                self.values.base[:,i,:]         = self.values.base[:,ny+i,:]
                self.values.base[:,ng+ny+1+i,:] = self.values.base[:,ng+i+1,:]
                # half
                self.values.base[i,:,:]       = self.values.base[nx+i,:,:]
                self.values.base[ng+nx+i,:,:] = self.values.base[ng+i,:,:]

        elif self.name == 'QT' or self.name == 'T':
            for i in range(ng):
                # half
                self.values.base[i,:,:]       = self.values.base[nx+i,:,:]
                self.values.base[ng+nx+i,:,:] = self.values.base[ng+i,:,:]
                self.values.base[:,i,:]       = self.values.base[:,ny+i,:]
                self.values.base[:,ng+ny+i,:] = self.values.base[:,ng+i,:]

        elif self.name == 'P':
            for i in range(ng):
                # half
                self.values.base[i,:,nl]       = self.values.base[nx+i,:,nl]
                self.values.base[ng+nx+i,:,nl] = self.values.base[ng+i,:,nl]
                self.values.base[:,i,nl]      = self.values.base[:,ny+i,nl]
                self.values.base[:,ng+ny+i,nl] = self.values.base[:,ng+i,nl]
        return

cdef class PrognosticVariables:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        cdef:
            Py_ssize_t ng = Gr.ng
        self.U  = PrognosticVariable(Gr.nx+2*ng+1, Gr.ny+2*ng,   Gr.nl,   'zonal velocity'      ,'u'  ,'1/s')
        self.V  = PrognosticVariable(Gr.nx+2*ng,   Gr.ny+2*ng+1, Gr.nl,   'meridionla velocity' ,'v'  ,'1/s')
        self.T  = PrognosticVariable(Gr.nx+2*ng,   Gr.ny+2*ng,   Gr.nl,   'Temperature'         ,'T'  ,'K')
        self.QT = PrognosticVariable(Gr.nx+2*ng,   Gr.ny+2*ng,   Gr.nl,   'Specific Humidity'   ,'QT' ,'K')
        self.P  = PrognosticVariable(Gr.nx+2*ng,   Gr.ny+2*ng,   Gr.nl+1, 'Pressure'            ,'p'  ,'pasc')
        return

    cpdef initialize(self, Parameters Pr):
        self.P_init = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_U')
        Stats.add_global_mean('global_mean_V')
        Stats.add_global_mean('global_mean_T')
        Stats.add_global_mean('global_mean_QT')
        Stats.add_axisymmetric_mean('axisymmetric_mean_U')
        Stats.add_axisymmetric_mean('axisymmetric_mean_V')
        Stats.add_axisymmetric_mean('axisymmetric_mean_T')
        Stats.add_axisymmetric_mean('axisymmetric_mean_QT')
        return

    # quick utility to set arrays with values in the "new" arrays
    cpdef apply_bc(self, Parameters Pr, Grid Gr):
        self.U.set_bcs(Pr,Gr)
        self.V.set_bcs(Pr,Gr)
        self.T.set_bcs(Pr,Gr)
        self.QT.set_bcs(Pr,Gr)
        self.P.set_bcs(Pr,Gr)
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
        return
    # this should be done in time intervals and save each time new files,not part of stats

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_U', self.U.values)
        Stats.write_global_mean('global_mean_V', self.V.values)
        Stats.write_global_mean('global_mean_T', self.T.values)
        Stats.write_global_mean('global_mean_QT', self.QT.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_U',self.U.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_V',self.V.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_T',self.T.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_QT',self.QT.values)
        return

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t nl = Pr.n_layers
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'U',                 self.U.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'V',                 self.V.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'Temperature',       self.T.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'dQTdt',             self.QT.mp_tendency[:,:,0:nl])
        Stats.write_2D_variable(Pr, Gr, int(TS.t),    'Pressure',          self.P.values[:,:,nl])
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Gr.nx
            Py_ssize_t ny = Gr.ny
            Py_ssize_t nl = Gr.nl
            Py_ssize_t ng = Gr.ng
            double fu, fv, dFx_dx, dFy_dy
            double dwT_dp_dn, dwT_dp_up, dwQT_dp_dn, dwQT_dp_up
            # double wu_dn, wu_up, wv_dn, wv_up
            double Wp_dgZ_dp, T_sur_flux, QT_sur_flux, U_sur_flux, V_sur_flux
            double dxi = 1.0/Gr.dx
            double dyi = 1.0/Gr.dy
            double [:,:,:] dgZ_dx = np.zeros((nx+2*ng, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] dgZ_dy = np.zeros((nx+2*ng, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] duu_dx = np.zeros((nx+2*ng+1, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] dvu_dy = np.zeros((nx+2*ng+1, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] duv_dx = np.zeros((nx+2*ng, ny+2*ng+1, nl),dtype=np.float64, order='c')
            double [:,:,:] dvv_dy = np.zeros((nx+2*ng, ny+2*ng+1, nl),dtype=np.float64, order='c')
            double [:,:,:] wu_dn = np.zeros((nx+2*ng+1, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] wv_dn = np.zeros((nx+2*ng+1, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] wu_up = np.zeros((nx+2*ng, ny+2*ng+1, nl),dtype=np.float64, order='c')
            double [:,:,:] wv_up = np.zeros((nx+2*ng, ny+2*ng+1, nl),dtype=np.float64, order='c')

        # flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.U)
        # flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.V)
        # flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.T)
        # if Pr.moist_index > 0.0:
        #     flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.QT)
        # with nogil:
        # for T

        for i in range(ng,nx+ng):
            for j in range(ng,ny+ng):
                ps_u_m = 0.5*(PV.P.values[i,j,nl]+PV.P.values[i-1,j,nl])
                ps_u_p = 0.5*(PV.P.values[i,j,nl]+PV.P.values[i+1,j,nl])
                ps_v_m = 0.5*(PV.P.values[i,j,nl]+PV.P.values[i,j-1,nl])
                ps_v_p = 0.5*(PV.P.values[i,j,nl]+PV.P.values[i,j+1,nl])
                dpudx = (PV.U.values[i+1,j,nl-1]*(ps_u_p - PV.P.values[i,j,nl-1]) -
                         PV.U.values[i,j,nl-1]  *(ps_u_m - PV.P.values[i,j,nl-1]))
                dpvdx = (PV.V.values[i,j+1,nl-1]*(ps_v_p - PV.P.values[i,j,nl-1]) -
                         PV.V.values[i,j,nl-1]  *(ps_v_m - PV.P.values[i,j,nl-1]))

                PV.P.tendency[i,j,nl] = (DV.Wp.values[i,j,nl-1] - dpudx - dpvdx)

                # PV.P.tendency[i,j,nl] = (DV.Wp.values[i,j,nl-1]
                #         + (PV.P.values[i,j,nl-1]-PV.P.values[i,j,nl])*DV.Divergence.values[i,j,nl-1]
                #    - 0.25*(PV.U.values[i+1,j,nl-1]+PV.U.values[i,j,nl-1])*(PV.P.values[i+1,j,nl]-PV.P.values[i-1,j,nl])*dxi
                #    - 0.25*(PV.V.values[i,j+1,nl-1]+PV.V.values[i,j,nl-1])*(PV.P.values[i,j+1,nl]-PV.P.values[i,j-1,nl])*dyi)
                for k in range(nl):
                    if k==nl-1:
                        T_sur_flux  = PV.T.SurfaceFlux[i,j]
                        QT_sur_flux = PV.QT.SurfaceFlux[i,j]
                    Wp_dgZ_dp = ((DV.Wp.values[i,j,k+1] + DV.Wp.values[i,j,k])/2.0/Pr.cp*
                                 (DV.gZ.values[i,j,k+1] - DV.gZ.values[i,j,k])
                                /(PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                    # Wp_dgZ_dp = (DV.Wp.values[i,j,k+1]/Pr.cp*
                    #     (DV.gZ.values[i,j,k+1]-DV.gZ.values[i,j,k])
                    #    /(PV.P.values[i,j,k+1]-PV.P.values[i,j,k]))
                    duT_dx = 0.5*((PV.T.values[i+1,j,k]+ PV.T.values[i,j,k])   * PV.U.values[i+1,j,k]-
                                  (PV.T.values[i,j,k]  + PV.T.values[i-1,j,k]) * PV.U.values[i-1,j,k])*dxi
                    dvT_dy = 0.5*((PV.T.values[i,j+1,k]+ PV.T.values[i,j,k])   * PV.V.values[i,j+1,k]-
                                  (PV.T.values[i,j,k]  + PV.T.values[i,j-1,k]) * PV.V.values[i,j-1,k])*dyi
                    if k==0:
                        dwT_dp_dn = 0.5*DV.Wp.values[i,j,k+1]*((PV.T.values[i,j,k+1] + PV.T.values[i,j,k])/
                                                                (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                        dwT_dp_up =  0.0
                    elif k==nl-1:
                        dwT_dp_dn = 0.5*DV.Wp.values[i,j,k+1]*((PV.T.values[i,j,k]  + PV.T.values[i,j,k])/
                                                               (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                        dwT_dp_up = 0.5*DV.Wp.values[i,j,k]*((PV.T.values[i,j,k]    + PV.T.values[i,j,k-1])/
                                                              (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                    else:
                        dwT_dp_dn = 0.5*DV.Wp.values[i,j,k+1]*((PV.T.values[i,j,k+1] + PV.T.values[i,j,k])/
                                                               (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                        dwT_dp_up = 0.5*DV.Wp.values[i,j,k]*((PV.T.values[i,j,k]    + PV.T.values[i,j,k-1])/
                                                             (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))

                    PV.T.tendency[i,j,k]  = ( - duT_dx - dvT_dy + dwT_dp_dn - dwT_dp_up - Wp_dgZ_dp + PV.T.mp_tendency[i,j,k]
                                             + PV.T.forcing[i,j,k] + T_sur_flux)

                    if Pr.moist_index > 0.0:
                        if k==0:
                            dwQT_dp_dn = 0.5*DV.Wp.values[i,j,k+1]*((PV.QT.values[i,j,k+1] + PV.QT.values[i,j,k])/
                                                                    (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                            dwQT_dp_up =  0.0
                        elif k==nl-1:
                            dwQT_dp_dn = 0.5*DV.Wp.values[i,j,k+1]*((PV.QT.values[i,j,k]  + PV.QT.values[i,j,k])/
                                                                   (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                            dwQT_dp_up = 0.5*DV.Wp.values[i,j,k]*((PV.QT.values[i,j,k]    + PV.QT.values[i,j,k-1])/
                                                                  (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                        else:
                            dwQT_dp_dn = 0.5*DV.Wp.values[i,j,k+1]*((PV.QT.values[i,j,k+1] + PV.QT.values[i,j,k])/
                                                                   (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                            dwQT_dp_up = 0.5*DV.Wp.values[i,j,k]*( (PV.QT.values[i,j,k]    + PV.QT.values[i,j,k-1])/
                                                                   (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))

                        duQT_dx = 0.5*((PV.QT.values[i+1,j,k]+ PV.QT.values[i,j,k])   * PV.U.values[i+1,j,k]-
                                       (PV.QT.values[i,j,k]  + PV.QT.values[i-1,j,k]) * PV.U.values[i-1,j,k])*dxi
                        dvQT_dy = 0.5*((PV.QT.values[i,j+1,k]+ PV.QT.values[i,j,k])   * PV.V.values[i,j+1,k]-
                                       (PV.QT.values[i,j,k]  + PV.QT.values[i,j-1,k]) * PV.V.values[i,j-1,k])*dyi
                        PV.QT.tendency[i,j,k]  = (- duQT_dx - dvQT_dy + dwQT_dp_dn - dwQT_dp_up + PV.QT.mp_tendency[i,j,k] + QT_sur_flux 
                                                  + PV.QT.HyperDiffusion[i,j,k])




        # for u
        for i in range(ng,nx+1+ng):
            for j in range(ng,ny+ng):
                for k in range(nl):
                    if k==nl-1:
                        U_sur_flux  = PV.U.SurfaceFlux[i,j]

                    fv = Pr.Coriolis*(PV.V.values[i-1,j,k]+PV.V.values[i-1,j+1,k]+
                              PV.V.values[i,j,k]+PV.V.values[i,j+1,k])*0.25
                    dgZ_dx[i,j,k] = ((DV.gZ.values[i,j,k]+DV.gZ.values[i,j,k+1])
                             -(DV.gZ.values[i-1,j,k]+DV.gZ.values[i-1,j,k+1]))*dxi/2.0
                    duu_dx[i,j,k] = 0.5*(PV.U.values[i+1,j,k] * PV.U.values[i+1,j,k]-
                                  PV.U.values[i-1,j,k] * PV.U.values[i-1,j,k])*dxi
                    dvu_dy[i,j,k] = 0.25*(
                                         (PV.U.values[i,j+1,k] + PV.U.values[i,j,k]) *
                                         (PV.V.values[i,j+1,k] + PV.V.values[i-1,j+1,k])-
                                         (PV.U.values[i,j,k]   + PV.U.values[i,j-1,k]) *
                                         (PV.V.values[i,j,k]   + PV.V.values[i-1,j,k])
                                         )*dyi

                    w_k  = 0.25*(DV.Wp.values[i,j,k]   + DV.Wp.values[i-1,j,k] +
                                 DV.Wp.values[i,j-1,k] + DV.Wp.values[i-1,j-1,k])
                    w_kp = 0.25*(DV.Wp.values[i,j,k+1]   + DV.Wp.values[i-1,j,k+1] +
                                 DV.Wp.values[i,j-1,k+1] + DV.Wp.values[i-1,j-1,k+1])
                    if k==0:
                        wu_dn[i,j,k] = 0.5*DV.Wp.values[i,j,k+1]*((PV.U.values[i,j,k+1] + PV.U.values[i,j,k])/
                                                                (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                        wu_up[i,j,k] =  0.0
                    elif k==nl-1:
                        wu_dn[i,j,k] = 0.5*w_kp*((PV.U.values[i,j,k]  + PV.U.values[i,j,k])/
                                          (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                        wu_up[i,j,k] = 0.5*w_k*((PV.U.values[i,j,k]    + PV.U.values[i,j,k-1])/
                                         (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                    else:
                        wu_dn[i,j,k] = 0.5*w_kp*((PV.U.values[i,j,k+1] + PV.U.values[i,j,k])/
                                          (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                        wu_up[i,j,k] = 0.5*w_k*( (PV.U.values[i,j,k]    + PV.U.values[i,j,k-1])/
                                          (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))


                    PV.U.tendency[i,j,k]  = ( fv - dgZ_dx[i,j,k])
                    # PV.U.tendency[i,j,k]  = (- duu_dx[i,j,k] - dvu_dy[i,j,k] - wu_dn[i,j,k] - wu_up[i,j,k] + fv - dgZ_dx[i,j,k] + PV.U.HyperDiffusion[i,j,k])


        # for v
        for i in range(ng,nx+ng):
            for j in range(ng,ny+1+ng):
                for k in range(nl):
                    if k==nl-1:
                        V_sur_flux  = PV.V.SurfaceFlux[i,j]

                    fu = Pr.Coriolis*(PV.U.values[i,j-1,k]+
                                      PV.U.values[i+1,j-1,k]+
                                      PV.U.values[i,j,k]+
                                      PV.U.values[i+1,j,k])*0.25
                    dgZ_dy[i,j,k] = ((DV.gZ.values[i,j,k]+DV.gZ.values[i,j,k+1])
                             -(DV.gZ.values[i,j-1,k]+DV.gZ.values[i,j-1,k+1]))*dyi/2.0
                    duv_dx[i,j,k] = 0.25*(
                                         (PV.U.values[i+1,j,k] + PV.U.values[i+1,j-1,k]) *
                                         (PV.V.values[i,j,k]   + PV.V.values[i+1,j,k])-
                                         (PV.U.values[i,j,k]   + PV.U.values[i,j-1,k]) *
                                         (PV.V.values[i,j,k]   + PV.V.values[i-1,j,k])
                                         )*dxi

                    dvv_dy[i,j,k] = 0.5*(PV.V.values[i,j+1,k] * PV.V.values[i,j+1,k]-
                                  PV.V.values[i,j-1,k] * PV.V.values[i,j-1,k])*dyi

                    w_k  = 0.25*(DV.Wp.values[i,j,k]   + DV.Wp.values[i-1,j,k] +
                                 DV.Wp.values[i,j-1,k] + DV.Wp.values[i-1,j-1,k])
                    w_kp = 0.25*(DV.Wp.values[i,j,k+1]   + DV.Wp.values[i-1,j,k+1] +
                                 DV.Wp.values[i,j-1,k+1] + DV.Wp.values[i-1,j-1,k+1])
                    if k==0:
                        wv_dn[i,j,k] = 0.5*DV.Wp.values[i,j,k+1]*((PV.V.values[i,j,k+1] + PV.V.values[i,j,k])/
                                                                (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                        wv_up[i,j,k] =  0.0
                    elif k==nl-1:
                        wv_dn[i,j,k] = 0.5*w_kp*((PV.V.values[i,j,k]  + PV.V.values[i,j,k])/
                                          (PV.P.values[i,j,k+1] - PV.P.values[i,j,k]))
                        wv_up[i,j,k] = 0.5*w_k*((PV.V.values[i,j,k]    + PV.V.values[i,j,k-1])/
                                         (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                    else:
                        wv_dn[i,j,k] = 0.5*w_kp*((PV.V.values[i,j,k+1] + PV.V.values[i,j,k])/
                                          (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))
                        wv_up[i,j,k] = 0.5*w_k*( (PV.V.values[i,j,k]    + PV.V.values[i,j,k-1])/
                                          (PV.P.values[i,j,k+1]  - PV.P.values[i,j,k]))

                    PV.V.tendency[i,j,k]  = (  - fu  - dgZ_dy[i,j,k])
                    # PV.V.tendency[i,j,k]  = (- duv_dx[i,j,k] - dvv_dy[i,j,k] - wv_dn[i,j,k] - wv_up[i,j,k] - fu  - dgZ_dy[i,j,k] + PV.V.HyperDiffusion[i,j,k])

        import pylab as plt
        plt.figure('gZ')
        plt.contourf(DV.gZ.values[:,:,1])
        plt.colorbar()
        plt.figure('V tendency')
        plt.contourf(PV.V.tendency[:,:,1])
        plt.colorbar()
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        return