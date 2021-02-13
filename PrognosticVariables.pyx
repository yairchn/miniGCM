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
    void rhs_qt(double* qt, double* u, double* v,double* qt_mp, double* qt_sur,
                double* rhs_qt, double* u_qt, double* v_qt, Py_ssize_t imax,
                Py_ssize_t jmax, Py_ssize_t kmax, Py_ssize_t k) nogil

    void rhs_T(double* H, double* u, double* v, double* H_mp, double* H_sur,
               double* H_forc, double* rhs_H, double* u_H, double* v_H,
               Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax, Py_ssize_t k) nogil

    void RHS_momentum(double g, double* H, double* vort, double* f,double* u, double* v,
                      double* ke, double* e_dry, double* u_vort, double* v_vort,
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
        if name=='H':
            self.forcing = np.zeros((nx,ny,nl),dtype = np.float64, order='c')
        if name=='H' or name=='QT':
            self.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            self.SurfaceFlux = np.zeros((nx,ny)   ,dtype=np.float64, order='c')


        return

cdef class PrognosticVariables:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.Vorticity   = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity'         ,'zeta' ,'1/s')
        self.Divergence  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Divergance'        ,'delta','1/s')
        self.H           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'       ,'T'    ,'K')
        self.QT          = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Specific Humidity' ,'QT'   ,'K')
        return

    cpdef initialize(self, Parameters Pr):
        self.H_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_H')
        Stats.add_global_mean('global_mean_QT')
        Stats.add_zonal_mean('zonal_mean_P')
        Stats.add_zonal_mean('zonal_mean_H')
        Stats.add_zonal_mean('zonal_mean_QT')
        Stats.add_zonal_mean('zonal_mean_divergence')
        Stats.add_zonal_mean('zonal_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_divergence')
        Stats.add_meridional_mean('meridional_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_P')
        Stats.add_meridional_mean('meridional_mean_H')
        Stats.add_meridional_mean('meridional_mean_QT')
        return

    # convert spherical data to spectral
    # I needto define this function to ast on a general variable
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t nl = Pr.n_layers
        for k in range(nl):
            self.Vorticity.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Vorticity.values.base[:,:,k])
            self.Divergence.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values.base[:,:,k])
            self.H.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.H.values.base[:,:,k])
            self.QT.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.QT.values.base[:,:,k])
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef spectral_to_physical(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t nl = Pr.n_layers

        for k in range(nl):
            self.Vorticity.values.base[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral.base[:,k])
            self.Divergence.values.base[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral.base[:,k])
            self.H.values.base[:,:,k]          = Gr.SphericalGrid.spectogrd(self.H.spectral.base[:,k])
            self.QT.values.base[:,:,k]         = Gr.SphericalGrid.spectogrd(self.QT.spectral.base[:,k])
        return

    # quick utility to set arrays with values in the "new" arrays
    cpdef set_old_with_now(self):
        self.Vorticity.old  = self.Vorticity.now.copy()
        self.Divergence.old = self.Divergence.now.copy()
        self.H.old          = self.H.now.copy()
        self.QT.old         = self.QT.now.copy()
        return

    cpdef set_now_with_tendencies(self):
        self.Vorticity.now  = self.Vorticity.tendency.copy()
        self.Divergence.now = self.Divergence.tendency.copy()
        self.H.now          = self.H.tendency.copy()
        self.QT.now         = self.QT.tendency.copy()
        return


    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_T', self.T.values)
        Stats.write_global_mean('global_mean_QT', self.QT.values)
        Stats.write_zonal_mean('zonal_mean_T',self.T.values)
        Stats.write_zonal_mean('zonal_mean_QT',self.QT.values)
        Stats.write_zonal_mean('zonal_mean_divergence',self.Divergence.values)
        Stats.write_zonal_mean('zonal_mean_vorticity',self.Vorticity.values)
        Stats.write_meridional_mean('meridional_mean_T',self.T.values)
        Stats.write_meridional_mean('meridional_mean_QT',self.QT.values)
        Stats.write_meridional_mean('meridional_mean_divergence',self.Divergence.values)
        Stats.write_meridional_mean('meridional_mean_vorticity',self.Vorticity.values)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t nl = Pr.n_layers
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Depth',             self.H.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'dQTdt',             self.QT.mp_tendency[:,:,0:nl])
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
            double complex [:] Vort_forc = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Div_forc  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] RHS_H     = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] RHS_QT    = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vort_sur_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Div_sur_flux  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_H_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Divergent_H_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_QT_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_P_flux  = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Divergent_QT_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Dry_Energy_laplacian = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Vortical_momentum_flux = np.zeros((nlm), dtype = np.complex, order ='c')
            double complex [:] Divergent_momentum_flux = np.zeros((nlm), dtype = np.complex, order ='c')

            double [:,:] uH  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] vH  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] uQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] vQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] Dry_Energy = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] u_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] v_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] RHS_grid_H  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] RHS_grid_QT = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] H_sur_flux  = np.zeros((nx, ny),dtype=np.float64, order ='c')
            double [:,:] QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')


        for k in range(nl):
            if k==nl-1:
                Vort_sur_flux ,Div_sur_flux = Gr.SphericalGrid.getvrtdivspec(DV.U.SurfaceFlux.base, DV.V.SurfaceFlux.base)
                H_sur_flux  = PV.H.SurfaceFlux
                QT_sur_flux = PV.QT.SurfaceFlux

            with nogil:
                RHS_momentum(double Pr.g, &DV.H.values[0,0,0], &PV.Vorticity.values[0,0,0], &Gr.Coriolis[0,0],
                            &DV.U.values[0,0,0], &DV.V.values[0,0,0], &DV.KE.values[0,0,0], &Dry_Energy[0,0],
                            &u_vorticity[0,0], &v_vorticity[0,0], nx, ny, nl, k)

                rhs_T(&PV.H.values[0,0,0], &DV.U.values[0,0,0], &DV.V.values[0,0,0], &PV.H.mp_tendency[0,0,0],
                      &H_sur_flux[0,0], &PV.H.forcing[0,0,0], &RHS_grid_H[0,0], &uH[0,0], &vH[0,0], nx, ny, nl, k)

            if Pr.moist_index > 0.0:
                rhs_qt(&PV.QT.values[0,0,0], &DV.U.values[0,0,0], &DV.V.values[0,0,0],
                       &PV.QT.mp_tendency[0,0,0], &QT_sur_flux[0,0], &RHS_grid_QT[0,0],
                       &uQT[0,0], &vQT[0,0], nx, ny, nl, k)

            Dry_Energy_laplacian = Gr.laplacian*Gr.SphericalGrid.grdtospec(Dry_Energy.base)
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vorticity.base, v_vorticity.base)
            Vortical_H_flux, Divergent_H_flux = Gr.SphericalGrid.getvrtdivspec(uH.base, vH.base) # Vortical_H_flux is not used
            Vort_forc ,Div_forc = Gr.SphericalGrid.getvrtdivspec(DV.U.forcing.base[:,:,k],DV.V.forcing.base[:,:,k])
            RHS_T  = Gr.SphericalGrid.grdtospec(RHS_grid_T.base)

            for i in range(nlm):
                Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(uQT.base, vQT.base) # Vortical_T_flux is not used
                RHS_QT = Gr.SphericalGrid.grdtospec(RHS_grid_QT.base)

            with nogil:
                for i in range(nlm):
                    PV.Vorticity.tendency[i,k]  = (Vort_forc[i] - Divergent_momentum_flux[i]
                                                    - w_vort_up[i] - w_vort_dn[i] + Vort_sur_flux[i])
                    PV.Divergence.tendency[i,k] = (Vortical_momentum_flux[i] - Dry_Energy_laplacian[i]
                                                    - w_div_up[i] - w_div_dn[i] + Div_forc[i] + Div_sur_flux[i])

                    PV.H.tendency[i,k]  = RHS_H[i]  - Divergent_H_flux[i]

                if Pr.moist_index > 0.0:
                    PV.QT.tendency[i,k] = RHS_QT[i] - Divergent_QT_flux[i]
        return
