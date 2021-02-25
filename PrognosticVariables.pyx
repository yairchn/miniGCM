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

        with nogil:
            for k in range(nl):
                for i in range(nx):
                    for j in range(0,ny):
                        q = ny-ng+j
                        self.values[i,j,k] = self.values[i,q,k]
                        self.values[i,j,k] = self.values[i,q,k]
                        self.values[j,i,k] = self.values[q,i,k]
                        self.values[j,i,k] = self.values[q,i,k]
                        q = ny+j+1
                        p = ng+j
                        self.values[i,q,k] = self.values[i,p,k]
                        self.values[i,q,k] = self.values[i,p,k]
                        self.values[q,i,k] = self.values[p,i,k]
                        self.values[q,i,k] = self.values[p,i,k]
        return

cdef class PrognosticVariables:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.U  = PrognosticVariable(Gr.nx, Gr.ny, Gr.nl,   'zonal velocity'      ,'u'  ,'1/s')
        self.V  = PrognosticVariable(Gr.nx, Gr.ny, Gr.nl,   'meridionla velocity' ,'v'  ,'1/s')
        self.T  = PrognosticVariable(Gr.nx, Gr.ny, Gr.nl,   'Temperature'         ,'T'  ,'K')
        self.QT = PrognosticVariable(Gr.nx, Gr.ny, Gr.nl,   'Specific Humidity'   ,'QT' ,'K')
        self.P  = PrognosticVariable(Gr.nx, Gr.ny, Gr.nl+1, 'Pressure'            ,'p'  ,'pasc')
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
            double fu, fv, dFx_dx, dFy_dy, dgZ_dx ,dgZ_dy
            double Wp_dgZ_dp, T_sur_flux, QT_sur_flux, U_sur_flux, V_sur_flux
            double dxi = 1.0/Gr.dx
            double dyi = 1.0/Gr.dy

        flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.U)
        flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.V)
        flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.T)
        if Pr.moist_index > 0.0:
            flux_constructor_fv(Pr, Gr, PV.U, PV.V, PV.QT)

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

                        fv = Pr.Coriolis*PV.V.values[i,j,k]
                        fu = Pr.Coriolis*PV.U.values[i,j,k]
                        dgZ_dx = (DV.gZ.values[i+1,j,k]-DV.gZ.values[i-1,j,k])*dxi
                        dgZ_dy = (DV.gZ.values[i,j+1,k]-DV.gZ.values[i,j-1,k])*dyi
                        Wp_dgZ_dp = ((DV.Wp.values[i,j,k+1]+DV.Wp.values[i,j,k])/2.0*
                            (DV.gZ.values[i,j,k+1]-DV.gZ.values[i,j,k-1])
                           /(PV.P.values[i,j,k+1]-PV.P.values[i,j,k-1]))

                        dFx_dx = 0.5*(PV.U.ZonalFlux[i+1,j,k]-PV.U.ZonalFlux[i-1,j,k])*dxi
                        dFy_dy = 0.5*(PV.U.ZonalFlux[i,j+1,k]-PV.U.ZonalFlux[i,j-1,k])*dyi
                        PV.U.tendency[i,j,k]  = - dFx_dx - dFy_dy - fv -dgZ_dx

                        dFx_dx = 0.5*(PV.V.ZonalFlux[i+1,j,k]-PV.V.ZonalFlux[i-1,j,k])*dxi
                        dFy_dy = 0.5*(PV.V.ZonalFlux[i,j+1,k]-PV.V.ZonalFlux[i,j-1,k])*dyi
                        PV.V.tendency[i,j,k]  = - dFx_dx - dFy_dy + fu -dgZ_dy

                        dFx_dx = 0.5*(PV.T.ZonalFlux[i+1,j,k]-PV.T.ZonalFlux[i-1,j,k])*dxi
                        dFy_dy = 0.5*(PV.T.ZonalFlux[i,j+1,k]-PV.T.ZonalFlux[i,j-1,k])*dyi
                        PV.T.tendency[i,j,k]  = - dFx_dx - dFy_dy - Wp_dgZ_dp + PV.T.mp_tendency[i,j,k] + PV.T.forcing[i,j,k] + T_sur_flux

                        if Pr.moist_index > 0.0:
                            dFx_dx = 0.5*(PV.QT.ZonalFlux[i+1,j,k]-PV.QT.ZonalFlux[i-1,j,k])*dxi
                            dFy_dy = 0.5*(PV.QT.ZonalFlux[i,j+1,k]-PV.QT.ZonalFlux[i,j-1,k])*dyi
                            PV.QT.tendency[i,j,k]  = - dFx_dx - dFy_dy + PV.QT.mp_tendency[i,j,k] + QT_sur_flux
        return