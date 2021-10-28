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
                double* qt_mp, double* qt_sur, double* turbflux, double* rhs_qt, double* u_qt, double* v_qt,
                Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax, Py_ssize_t k) nogil

    void rhs_E(double cp, double* p, double* gz, double* T, double* u, double* v, double* wp,
               double* T_mp, double* E_sur, double* E_forc, double* turbflux, double* RHS_E, double* u_T,
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

        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.old      = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.now      = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.tendency = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.float64, order='c')
        self.ConvectiveFlux = np.zeros((n_spec,nl),dtype=np.complex, order='c')
        self.sp_VerticalFlux = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        if kind=='Vorticity':
            self.sp_forcing = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        if name=='T':
            self.forcing = np.zeros((nx,ny,nl),dtype = np.float64, order='c')
        if name=='T' or name=='QT':
            self.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            self.SurfaceFlux = np.zeros((nx,ny)   ,dtype=np.float64, order='c')
            self.TurbFlux    = np.zeros((nx,ny,nl),dtype=np.float64, order='c')


        return

cdef class PrognosticVariables:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.Vorticity   = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity'         ,'zeta' ,'1/s')
        self.Divergence  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Divergance'        ,'delta','1/s')
        self.E           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Total_Energy'       ,'m^2/s^2'    ,'K')
        self.QT          = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Specific Humidity' ,'QT'   ,'K')
        self.P           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,'p'    ,'pasc')
        return

    cpdef initialize(self, Parameters Pr):
        self.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        return

    cpdef initialize_io(self, Parameters Pr, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_E')
        Stats.add_zonal_mean('zonal_mean_E')
        Stats.add_zonal_mean('zonal_mean_divergence')
        Stats.add_zonal_mean('zonal_mean_vorticity')
        Stats.add_surface_zonal_mean('zonal_mean_Ps')
        Stats.add_surface_zonal_mean('E_SurfaceFlux')
        Stats.add_surface_zonal_mean('QT_SurfaceFlux')
        Stats.add_meridional_mean('meridional_mean_divergence')
        Stats.add_meridional_mean('meridional_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_E')
        Stats.add_surface_meridional_mean('meridional_mean_Ps')
        if Pr.moist_index > 0.0:
            Stats.add_global_mean('global_mean_QT')
            Stats.add_zonal_mean('zonal_mean_QT')
            Stats.add_meridional_mean('meridional_mean_QT')
            # Stats.add_global_mean('global_mean_dQTdt')
            # Stats.add_zonal_mean('zonal_mean_dQTdt')
            # Stats.add_meridional_mean('meridional_mean_dQTdt')
        return

    # convert spherical data to spectral
    # I needto define this function to ast on a general variable
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t nl = Pr.n_layers
        self.P.spectral.base[:,nl] = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,nl])
        for k in range(nl):
            self.Vorticity.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Vorticity.values.base[:,:,k])
            self.Divergence.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values.base[:,:,k])
            self.E.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.E.values.base[:,:,k])
            if Pr.moist_index > 0.0:
                self.QT.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.QT.values.base[:,:,k])
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef spectral_to_physical(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t nl = Pr.n_layers

        self.P.values.base[:,:,nl] = Gr.SphericalGrid.spectogrd(np.copy(self.P.spectral.base[:,nl]))
        for k in range(nl):
            self.Vorticity.values.base[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral.base[:,k])
            self.Divergence.values.base[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral.base[:,k])
            self.E.values.base[:,:,k]          = Gr.SphericalGrid.spectogrd(self.E.spectral.base[:,k])
            if Pr.moist_index > 0.0:
                self.QT.values.base[:,:,k]         = np.clip(Gr.SphericalGrid.spectogrd(self.QT.spectral.base[:,k]), 0.0, 1.0)
        return

    # quick uEility to set arrays with values in the "new" arrays
    cpdef set_old_with_now(self):
        self.Vorticity.old  = self.Vorticity.now.copy()
        self.Divergence.old = self.Divergence.now.copy()
        self.E.old          = self.E.now.copy()
        self.QT.old         = self.QT.now.copy()
        self.P.old          = self.P.now.copy()
        return

    cpdef set_now_with_tendencies(self):
        self.Vorticity.now  = self.Vorticity.tendency.copy()
        self.Divergence.now = self.Divergence.tendency.copy()
        self.E.now          = self.E.tendency.copy()
        self.QT.now         = self.QT.tendency.copy()
        self.P.now          = self.P.tendency.copy()
        return

    cpdef reset_pressures_and_bcs(self, Parameters Pr, DiagnosticVariables DV):
        cdef:
            Py_ssize_t nl = Pr.n_layers

        DV.gZ.values.base[:,:,nl] = np.zeros_like(self.P.values.base[:,:,nl])
        DV.Wp.values.base[:,:,0]  = np.zeros_like(self.P.values.base[:,:,nl])
        for k in range(nl):
            self.P.values.base[:,:,k] = np.add(np.zeros_like(self.P.values.base[:,:,k]),Pr.pressure_levels[k])
            self.P.spectral.base[:,k] = np.add(np.zeros_like(self.P.spectral.base[:,k]),Pr.pressure_levels[k])
        return
    # this should be done in time intervals and save each time new files,not part of stats

    cpdef stats_io(self, Parameters Pr, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t nl = Pr.n_layers

        Stats.write_global_mean('global_mean_E', self.E.values)
        Stats.write_surface_zonal_mean('zonal_mean_Ps',self.P.values[:,:,nl])
        Stats.write_surface_zonal_mean('E_SurfaceFlux', self.E.SurfaceFlux)
        Stats.write_surface_zonal_mean('QT_SurfaceFlux',self.QT.SurfaceFlux)
        Stats.write_zonal_mean('zonal_mean_E',self.E.values)
        Stats.write_zonal_mean('zonal_mean_divergence',self.Divergence.values)
        Stats.write_zonal_mean('zonal_mean_vorticity',self.Vorticity.values)
        # Stats.write_surface_meridional_mean('meridional_mean_Ps',self.P.values[:,:,-1])
        Stats.write_meridional_mean('meridional_mean_E',self.E.values)
        Stats.write_meridional_mean('meridional_mean_divergence',self.Divergence.values)
        Stats.write_meridional_mean('meridional_mean_vorticity',self.Vorticity.values)
        if Pr.moist_index > 0.0:
            Stats.write_global_mean('global_mean_QT', self.QT.values)
            Stats.write_zonal_mean('zonal_mean_QT',self.QT.values)
            Stats.write_meridional_mean('meridional_mean_QT',self.QT.values)
            # Stats.write_global_mean('global_mean_dQTdt', self.QT.mp_tendency)
            # Stats.write_zonal_mean('zonal_mean_dQTdt',self.QT.mp_tendency)
            # Stats.write_meridional_mean('meridional_mean_dQTdt',self.QT.mp_tendency)
        return

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm

        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Total_Energy',      self.E.values)
        Stats.write_2D_variable(Pr, int(TS.t),    'Pressure',          self.P.values[:,:,nl])
        Stats.write_2D_variable(Pr, int(TS.t),    'E_SurfaceFlux',     self.E.SurfaceFlux)
        # Stats.write_spectral_field(Pr, int(TS.t),nlm, nl, 'Vorticity_forcing', self.Vorticity.sp_forcing)
        # Stats.write_spectral_field(Pr, int(TS.t),nlm, nl, 'Vorticity_ConvectiveFlux', self.Vorticity.ConvectiveFlux)
        # Stats.write_spectral_field(Pr, int(TS.t),nlm, nl, 'Divergence_ConvectiveFlux', self.Divergence.ConvectiveFlux)
        # Stats.write_spectral_field(Pr, int(TS.t),nlm, nl, 'Temperature_ConvectiveFlux', self.E.ConvectiveFlux)
        # Stats.write_spectral_field(Pr, int(TS.t),nlm, nl, 'Specific_humidity_ConvectiveFlux', self.QT.ConvectiveFlux)
        if Pr.moist_index > 0.0:
            Stats.write_3D_variable(Pr, int(TS.t),nl, 'Specific_humidity', self.QT.values)
            Stats.write_3D_variable(Pr, int(TS.t),nl, 'dQTdt',             self.QT.mp_tendency[:,:,0:nl])
            Stats.write_2D_variable(Pr, int(TS.t),    'QT_SurfaceFlux',    self.QT.SurfaceFlux)
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm

            double complex [:] w_vort_up = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] w_vort_dn = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] w_div_up  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] w_div_dn  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vort_forc = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Div_forc  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] RHS_E     = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] RHS_QT    = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vort_sur_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Div_sur_flux  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_E_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Divergent_E_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_QT_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_P_flux  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Divergent_P_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Divergent_QT_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Dry_Energy_laplacian = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_momentum_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Divergent_momentum_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Wp3_spectral = np.zeros((nlm), dtype = np.complex, order ='c')

            double [:,:] uE  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] vE  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] uQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] vQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wu_dn = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wv_dn = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wu_up = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] wv_up = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] Dry_Energy = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] u_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] v_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] RHS_grid_E  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] RHS_grid_QT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] E_sur_flux  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] u_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] v_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')


        Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(
            np.multiply(DV.U.values[:,:,nl-1],np.subtract(PV.P.values[:,:,nl-1],PV.P.values[:,:,nl])),
            np.multiply(DV.V.values[:,:,nl-1],np.subtract(PV.P.values[:,:,nl-1],PV.P.values[:,:,nl])))

        Wp3_spectral = Gr.SphericalGrid.grdtospec(DV.Wp.values.base[:,:,nl-1])
        with nogil:
            for i in range(nlm):
                PV.P.tendency[i,nl] = Divergent_P_flux[i] + Wp3_spectral[i]

        for k in range(nl):
            if k==nl-1:
                Vort_sur_flux ,Div_sur_flux = Gr.SphericalGrid.getvrtdivspec(DV.U.SurfaceFlux.base, DV.V.SurfaceFlux.base)
                E_turb_flux  = PV.E.SurfaceFlux
                QE_turb_flux = PV.QT.SurfaceFlux

            with nogil:
                vertical_uv_fluxes(&PV.P.values[0,0,0], &DV.gZ.values[0,0,0], &PV.Vorticity.values[0,0,0],
                            &Gr.Coriolis[0,0], &DV.U.values[0,0,0], &DV.V.values[0,0,0],
                            &DV.Wp.values[0,0,0], &DV.KE.values[0,0,0], &wu_up[0,0], &wv_up[0,0], &wu_dn[0,0], &wv_dn[0,0],
                            &Dry_Energy[0,0], &u_vorticity[0,0], &v_vorticity[0,0],
                            nx, ny, nl, k)

                rhs_E(Pr.cp, &PV.P.values[0,0,0], &DV.gZ.values[0,0,0], &PV.E.values[0,0,0], &DV.U.values[0,0,0],
                           &DV.V.values[0,0,0], &DV.Wp.values[0,0,0], &PV.E.mp_tendency[0,0,0], &E_sur_flux[0,0],
                           &PV.E.forcing[0,0,0], &PV.E.TurbFlux[0,0,0], &RHS_grid_E[0,0], &uE[0,0], &vE[0,0], nx, ny, nl, k)

                if Pr.moist_index > 0.0:
                    rhs_qt(&PV.P.values[0,0,0], &PV.QT.values[0,0,0], &DV.U.values[0,0,0], &DV.V.values[0,0,0],
                           &DV.Wp.values[0,0,0], &PV.QT.mp_tendency[0,0,0], &QT_sur_flux[0,0], &PV.QT.TurbFlux[0,0,0],
                           &RHS_grid_QT[0,0], &uQT[0,0], &vQT[0,0], nx, ny, nl, k)

            Dry_Energy_laplacian = Gr.laplacian*Gr.SphericalGrid.grdtospec(Dry_Energy.base)
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vorticity.base, v_vorticity.base)
            Vortical_E_flux, Divergent_E_flux = Gr.SphericalGrid.getvrtdivspec(uE.base, vE.base) # Vortical_E_flux is not used
            Vort_forc ,Div_forc = Gr.SphericalGrid.getvrtdivspec(DV.U.forcing.base[:,:,k],DV.V.forcing.base[:,:,k])
            w_vort_up ,w_div_up = Gr.SphericalGrid.getvrtdivspec(wu_up.base, wv_up.base)
            w_vort_dn ,w_div_dn = Gr.SphericalGrid.getvrtdivspec(wu_dn.base, wv_dn.base)
            RHS_E  = Gr.SphericalGrid.grdtospec(RHS_grid_E.base)

            if Pr.moist_index > 0.0:
                Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(uQT.base, vQT.base) # Vortical_E_flux is not used
                RHS_QT = Gr.SphericalGrid.grdtospec(RHS_grid_QT.base)

            with nogil:
                for i in range(nlm):
                    PV.Vorticity.tendency[i,k]  = (Vort_forc[i] - Divergent_momentum_flux[i]- w_vort_up[i] - w_vort_dn[i]
                                                     + Vort_sur_flux[i] + PV.Vorticity.ConvectiveFlux[i,k] + PV.Vorticity.sp_forcing[i,k])
                    PV.Divergence.tendency[i,k] = (Vortical_momentum_flux[i] - Dry_Energy_laplacian[i]- w_div_up[i] - w_div_dn[i]
                                                    + Div_forc[i] + Div_sur_flux[i] + PV.Divergence.ConvectiveFlux[i,k])

                    PV.E.tendency[i,k]  = RHS_E[i]  - Divergent_E_flux[i] + PV.E.ConvectiveFlux[i,k]

                    if Pr.moist_index > 0.0:
                        PV.QT.tendency[i,k] = RHS_QT[i] - Divergent_QT_flux[i] + PV.QT.ConvectiveFlux[i,k]
        return
