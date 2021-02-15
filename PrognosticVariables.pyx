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
    def __init__(self, nx, ny, nl, n_spec, kind, name, units):
        self.kind = kind
        self.name = name
        self.units = units

        self.values   = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.old      = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.now      = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.float64, order='c')
        if name=='T':
            self.forcing = np.zeros((nx,ny,nl),dtype = np.float64, order='c')
        if name=='T' or name=='QT':
            self.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            self.SurfaceFlux = np.zeros((nx,ny)   ,dtype=np.float64, order='c')


        return

cdef class PrognosticVariables:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.U  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'zonal velocity'      ,'u'  ,'1/s')
        self.V  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'meridionla velocity' ,'v'  ,'1/s')
        self.T  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'         ,'T'  ,'K')
        self.QT = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Specific Humidity'   ,'QT' ,'K')
        self.P  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'            ,'p'  ,'pasc')
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
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm

            double [:] w_vort_up = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] w_vort_dn = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] w_div_up  = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] w_div_dn  = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Vort_forc = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Div_forc  = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] RHS_T     = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] RHS_QT    = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Vort_sur_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Div_sur_flux  = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Vortical_T_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Divergent_T_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Vortical_QT_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Vortical_P_flux  = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Divergent_P_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Divergent_QT_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Dry_Energy_laplacian = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Vortical_momentum_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')
            double [:] Divergent_momentum_flux = np.zeros((nx, ny), dtype = np.float64, order ='c')

            double [:,:] uT  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] vT  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] uQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] vQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wu_dn = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wv_dn = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wu_up = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wv_up = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] Dry_Energy = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] u_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] v_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] RHS_grid_T  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] RHS_grid_QT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] T_sur_flux  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] u_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] v_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')


        with nogil:
            for i in range(Pr.nx):
                for j in range(Pr.ny):
                    PV.P.tendency[i,j,nl] = DV.Wp.values[i,j,nl-1] + mass_divergence?
                    for k in range(nl):
                    if k==nl-1:
                        U_sur_flux  = DV.U.SurfaceFlux
                        V_sur_flux  = DV.V.SurfaceFlux
                        T_sur_flux  = PV.T.SurfaceFlux
                        QT_sur_flux = PV.QT.SurfaceFlux

                        duu_dx = 
                        duv_dy = 
                        duv_dx = 
                        dvv_dy = 
                        fv = Pr.Coriolis*PV.V.values[i,j,k]
                        fu = Pr.Coriolis*PV.U.values[i,j,k]
                        dgZ_dx = 
                        dgZ_dy = 
                        duT_dx = 
                        dvT_dy = 
                        Wp_dgZ_dp = 
                        duQT_dx = 
                        dvQT_dy = 
                        PV.U.tendency[i,j,k]  = - duu_dx - duv_dy - fv -dgZ_dx
                        PV.V.tendency[i,j,k]  = - duv_dx - dvv_dy + fu -dgZ_dy

                        PV.T.tendency[i,j,k]  = - duT_dx - dvT_dy -Wp_dgZ_dp + PV.T.mp_tendency[i,j,k] + PV.T.forcing[i,j,k] + T_sur_flux

                        if Pr.moist_index > 0.0:
                            PV.T.tendency[i,j,k]  = - duQT_dx - dvQT_dy + PV.QT.mp_tendency[i,j,k] + QT_sur_flux
        return
