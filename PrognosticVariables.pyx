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
        self.H  = PrognosticVariable(Gr.nx+2*ng,   Gr.ny+2*ng,   Gr.nl,   'Depth'               ,'H'  ,'K')
        self.QT = PrognosticVariable(Gr.nx+2*ng,   Gr.ny+2*ng,   Gr.nl,   'Specific Humidity'   ,'QT' ,'K')
        return

    cpdef initialize(self, Parameters Pr):
        self.H_init = np.array([Pr.H1, Pr.H2, Pr.H3])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_U')
        Stats.add_global_mean('global_mean_V')
        Stats.add_global_mean('global_mean_H')
        Stats.add_global_mean('global_mean_QT')
        Stats.add_axisymmetric_mean('axisymmetric_mean_U')
        Stats.add_axisymmetric_mean('axisymmetric_mean_V')
        Stats.add_axisymmetric_mean('axisymmetric_mean_H')
        Stats.add_axisymmetric_mean('axisymmetric_mean_QT')
        return

    # quick utility to set arrays with values in the "new" arrays
    cpdef apply_bc(self, Parameters Pr, Grid Gr):
        self.U.set_bcs(Pr,Gr)
        self.V.set_bcs(Pr,Gr)
        self.H.set_bcs(Pr,Gr)
        self.QT.set_bcs(Pr,Gr)
        return

    # quick utility to set arrays with values in the "new" arrays
    cpdef set_old_with_now(self):
        self.U.old  = self.U.now.copy()
        self.V.old  = self.V.now.copy()
        self.H.old  = self.H.now.copy()
        self.QT.old = self.QT.now.copy()
        return

    cpdef set_now_with_tendencies(self):
        self.U.now  = self.U.tendency.copy()
        self.V.now  = self.V.tendency.copy()
        self.H.now  = self.H.tendency.copy()
        self.QT.now = self.QT.tendency.copy()
        return

    # this should be done in time intervals and save each time new files,not part of stats

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_U', self.U.values)
        Stats.write_global_mean('global_mean_V', self.V.values)
        Stats.write_global_mean('global_mean_H', self.H.values)
        Stats.write_global_mean('global_mean_QT', self.QT.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_U',self.U.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_V',self.V.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_H',self.H.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_QT',self.QT.values)
        return

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t nl = Pr.n_layers
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'U',                 self.U.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'V',                 self.V.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'Depth',             self.H.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t),nl, 'dQTdt',             self.QT.mp_tendency[:,:,0:nl])
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
            double T_sur_flux, QT_sur_flux, U_sur_flux, V_sur_flux
            double dxi = 1.0/Gr.dx
            double dyi = 1.0/Gr.dy
            double [:,:,:] dgZ_dx = np.zeros((nx+2*ng, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] dgZ_dy = np.zeros((nx+2*ng, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] duu_dx = np.zeros((nx+2*ng+1, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] dvu_dy = np.zeros((nx+2*ng+1, ny+2*ng, nl),dtype=np.float64, order='c')
            double [:,:,:] duv_dx = np.zeros((nx+2*ng, ny+2*ng+1, nl),dtype=np.float64, order='c')
            double [:,:,:] dvv_dy = np.zeros((nx+2*ng, ny+2*ng+1, nl),dtype=np.float64, order='c')

        # flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.U)
        # flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.V)
        # flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.T)
        # if Pr.moist_index > 0.0:
        #     flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.QT)
        # with nogil:
        # for T

        for i in range(ng,nx+ng):
            for j in range(ng,ny+ng):
                for k in range(nl):
                    H_im = 0.5*(PV.H.values[i,j,k]+PV.H.values[i-1,j,k])
                    H_ip = 0.5*(PV.H.values[i,j,k]+PV.H.values[i+1,j,k])
                    H_jm = 0.5*(PV.H.values[i,j,k]+PV.H.values[i,j-1,k])
                    H_jp = 0.5*(PV.H.values[i,j,k]+PV.H.values[i,j+1,k])
                    duHdx = 0.5*(PV.U.values[i+1,j,k-1]*(PV.H.values[i,j,k]+PV.H.values[i+1,j,k]) -
                                 PV.U.values[i,j,k-1]  *(PV.H.values[i,j,k]+PV.H.values[i-1,j,k]))*dxi
                    dvHdy = 0.5*(PV.V.values[i,j+1,k-1]*(PV.H.values[i,j,k]+PV.H.values[i,j+1,k]) -
                                 PV.V.values[i,j,k-1]  *(PV.H.values[i,j,k]+PV.H.values[i,j+1,k]))*dyi

                    PV.H.tendency[i,j,nl] = - duHdx - dvHdy

                    if k==nl-1:
                        QT_sur_flux = PV.QT.SurfaceFlux[i,j]

                    if Pr.moist_index > 0.0:
                        duQT_dx = 0.5*((PV.QT.values[i+1,j,k]+ PV.QT.values[i,j,k])   * PV.U.values[i+1,j,k]-
                                       (PV.QT.values[i,j,k]  + PV.QT.values[i-1,j,k]) * PV.U.values[i,j,k])*dxi
                        dvQT_dy = 0.5*((PV.QT.values[i,j+1,k]+ PV.QT.values[i,j,k])   * PV.V.values[i,j+1,k]-
                                       (PV.QT.values[i,j,k]  + PV.QT.values[i,j-1,k]) * PV.V.values[i,j,k])*dyi
                        PV.QT.tendency[i,j,k]  = (- duQT_dx - dvQT_dy + PV.QT.mp_tendency[i,j,k] + QT_sur_flux
                                                  + PV.QT.HyperDiffusion[i,j,k])


        # for u
        for i in range(ng,nx+1+ng):
            for j in range(ng,ny+ng):
                for k in range(nl):
                    if k==nl-1:
                        U_sur_flux  = PV.U.SurfaceFlux[i,j]

                    fv = Pr.Coriolis*(PV.V.values[i-1,j,k]+PV.V.values[i-1,j+1,k]+
                              PV.V.values[i,j,k]+PV.V.values[i,j+1,k])*0.25
                    dgZ_dx[i,j,k] = Pr.g*((PV.H.values[i,j,k]+PV.H.values[i,j,k+1])
                             -(PV.H.values[i-1,j,k]+PV.H.values[i-1,j,k+1]))*dxi/2.0
                    duu_dx[i,j,k] = 0.5*(PV.U.values[i+1,j,k] * PV.U.values[i+1,j,k]-
                                  PV.U.values[i-1,j,k] * PV.U.values[i-1,j,k])*dxi
                    dvu_dy[i,j,k] = 0.25*(
                                         (PV.U.values[i,j+1,k] + PV.U.values[i,j,k]) *
                                         (PV.V.values[i,j+1,k] + PV.V.values[i-1,j+1,k])-
                                         (PV.U.values[i,j,k]   + PV.U.values[i,j-1,k]) *
                                         (PV.V.values[i,j,k]   + PV.V.values[i-1,j,k])
                                         )*dyi

                    PV.U.tendency[i,j,k]  = (- duu_dx[i,j,k] - dvu_dy[i,j,k] + fv - dgZ_dx[i,j,k] + PV.U.HyperDiffusion[i,j,k])


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
                    dgZ_dy[i,j,k] = Pr.g*((PV.H.values[i,j,k]+PV.H.values[i,j,k+1])
                             -(PV.H.values[i,j-1,k]+PV.H.values[i,j-1,k+1]))*dyi/2.0
                    duv_dx[i,j,k] = 0.25*(
                                         (PV.U.values[i+1,j,k] + PV.U.values[i+1,j-1,k]) *
                                         (PV.V.values[i,j,k]   + PV.V.values[i+1,j,k])-
                                         (PV.U.values[i,j,k]   + PV.U.values[i,j-1,k]) *
                                         (PV.V.values[i,j,k]   + PV.V.values[i-1,j,k])
                                         )*dxi

                    dvv_dy[i,j,k] = 0.5*(PV.V.values[i,j+1,k] * PV.V.values[i,j+1,k]-
                                  PV.V.values[i,j-1,k] * PV.V.values[i,j-1,k])*dyi

                    PV.V.tendency[i,j,k]  = (- duv_dx[i,j,k] - dvv_dy[i,j,k] - fu  - dgZ_dy[i,j,k] + PV.V.HyperDiffusion[i,j,k])

        # import pylab as plt
        # plt.figure('H')
        # plt.subplot(3,1,1)
        # plt.contourf(np.power(PV.V.values[:,1:,0],2.0)+np.power(PV.U.values[1:,:,0],2.0))
        # plt.colorbar()
        # plt.subplot(3,1,2)
        # plt.contourf(np.power(PV.V.values[:,1:,1],2.0)+np.power(PV.U.values[1:,:,1],2.0))
        # plt.colorbar()
        # plt.subplot(3,1,3)
        # plt.contourf(np.power(PV.V.values[:,1:,2],2.0)+np.power(PV.U.values[1:,:,2],2.0))
        # plt.colorbar()
        # plt.show(block=False)
        # plt.pause(0.3)
        # plt.close()
        return