import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import netCDF4
from NetCDFIO cimport NetCDFIO_Stats
import numpy as np
cimport numpy as np
import scipy as sc
import sys
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters
cimport Microphysics
import Microphysics

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
        self.sp_VerticalFlux = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        if name=='T':
            self.forcing = np.zeros((nx,ny,nl),dtype = np.float64, order='c')
        if name=='T' or name=='QT':
            self.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            self.SurfaceFlux = np.zeros((nx,ny)   ,dtype=np.float64, order='c')


        return

cdef class PrognosticVariables:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.Vorticity   = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity'         ,'zeta' ,'1/s')
        self.Divergence  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Divergance'        ,'delta','1/s')
        self.T           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'       ,'T'    ,'K')
        self.QT          = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Specific Humidity' ,'QT'   ,'K')
        self.P           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,'p'    ,'pasc')
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
        Stats.add_zonal_mean('zonal_mean_divergence')
        Stats.add_zonal_mean('zonal_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_divergence')
        Stats.add_meridional_mean('meridional_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_P')
        Stats.add_meridional_mean('meridional_mean_T')
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
            self.T.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.T.values.base[:,:,k])
            self.QT.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.QT.values.base[:,:,k])
        self.P.spectral.base[:,nl] = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,nl])
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
            self.T.values.base[:,:,k]          = Gr.SphericalGrid.spectogrd(self.T.spectral.base[:,k])
            self.QT.values.base[:,:,k]         = Gr.SphericalGrid.spectogrd(self.QT.spectral.base[:,k])
        self.P.values.base[:,:,nl] = Gr.SphericalGrid.spectogrd(np.copy(self.P.spectral.base[:,nl]))
        return

    # quick utility to set arrays with values in the "new" arrays
    cpdef set_old_with_now(self):
        self.Vorticity.old  = self.Vorticity.now.copy()
        self.Divergence.old = self.Divergence.now.copy()
        self.T.old          = self.T.now.copy()
        self.QT.old         = self.QT.now.copy()
        self.P.old          = self.P.now.copy()
        return

    cpdef set_now_with_tendencies(self):
        self.Vorticity.now  = self.Vorticity.tendency.copy()
        self.Divergence.now = self.Divergence.tendency.copy()
        self.T.now          = self.T.tendency.copy()
        self.QT.now         = self.QT.tendency.copy()
        self.P.now          = self.P.tendency.copy()
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
        Stats.write_zonal_mean('zonal_mean_divergence',self.Divergence.values)
        Stats.write_zonal_mean('zonal_mean_vorticity',self.Vorticity.values)
        Stats.write_meridional_mean('meridional_mean_P',self.P.values[:,:,1:4])
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
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Temperature',       self.T.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'dQTdt',             self.QT.mp_tendency[:,:,0:nl])
        Stats.write_2D_variable(Pr, int(TS.t),             'Pressure',          self.P.values[:,:,nl])
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm
            double dpi, wQT_dn, wQT_up, wT_dn, wT_up

            double complex [:] w_vort_up = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] w_vort_dn = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] w_div_up  = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] w_div_dn  = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vort_forc = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Div_forc  = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] RHS_T     = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] RHS_QT    = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vort_sur_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Div_sur_flux  = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vortical_T_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_T_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vortical_QT_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vortical_P_flux  = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_P_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_QT_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Dry_Energy_laplacian = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vortical_momentum_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_momentum_flux = np.zeros((nlm), dtype = np.complex, order='c')

            double [:,:] uT  = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] vT  = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] uQT = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] vQT = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] wu_dn = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] wv_dn = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] wu_up = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] wv_up = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] Dry_Energy = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] u_vorticity = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] v_vorticity = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] RHS_grid_T  = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] RHS_grid_QT = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] T_sur_flux  = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] u_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] v_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] Thermal_expension = np.zeros((nx, ny),dtype=np.float64, order='c')

        Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(
            np.multiply(DV.U.values[:,:,nl-1],np.subtract(PV.P.values[:,:,nl-1],PV.P.values[:,:,nl])),
            np.multiply(DV.V.values[:,:,nl-1],np.subtract(PV.P.values[:,:,nl-1],PV.P.values[:,:,nl])))

        PV.P.tendency.base[:,nl] = np.add(Divergent_P_flux, DV.Wp.spectral[:,nl-1])

        for k in range(nl):
            if k==nl-1:
                Vort_sur_flux ,Div_sur_flux = Gr.SphericalGrid.getvrtdivspec(DV.U.SurfaceFlux.base, DV.V.SurfaceFlux.base)
                T_sur_flux  = PV.T.SurfaceFlux
                QT_sur_flux = PV.QT.SurfaceFlux
            else:
                Vort_sur_flux = np.zeros((nlm), dtype = np.complex, order='c')
                Div_sur_flux = np.zeros((nlm), dtype = np.complex, order='c')
                T_sur_flux  = np.zeros((nx, ny),dtype=np.float64, order='c')
                QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            with nogil:
                for i in range(nx):
                    for j in range(ny):
                        dpi = 1.0/(PV.P.values[i,j,k+1] - PV.P.values[i,j,k])
                        Dry_Energy[i,j] = DV.gZ.values[i,j,k] + DV.KE.values[i,j,k]
                        u_vorticity[i,j] = DV.U.values[i,j,k] * (PV.Vorticity.values[i,j,k]+Gr.Coriolis[i,j])
                        v_vorticity[i,j] = DV.V.values[i,j,k] * (PV.Vorticity.values[i,j,k]+Gr.Coriolis[i,j])
                        uT[i,j]         = DV.U.values[i,j,k] * PV.T.values[i,j,k]
                        vT[i,j]         = DV.V.values[i,j,k] * PV.T.values[i,j,k]
                        uQT[i,j]        = DV.U.values[i,j,k] * PV.QT.values[i,j,k]
                        vQT[i,j]        = DV.V.values[i,j,k] * PV.QT.values[i,j,k]

                        if k==0:
                            wu_dn[i,j] = DV.Wp.values[i,j,k+1]*(DV.U.values[i,j,k+1] - DV.U.values[i,j,k])*dpi
                            wv_dn[i,j] = DV.Wp.values[i,j,k+1]*(DV.V.values[i,j,k+1] - DV.V.values[i,j,k])*dpi
                            wu_up[i,j] = 0.0
                            wv_up[i,j] = 0.0

                            wT_dn  = 0.5*DV.Wp.values[i,j,k+1]*(PV.T.values[i,j,k]+PV.T.values[i,j,k])*dpi
                            wQT_dn = 0.5*DV.Wp.values[i,j,k+1]*(PV.QT.values[i,j,k+1]+ PV.QT.values[i,j,k])*dpi
                            wT_up  = 0.0
                            wQT_up = 0.0

                        elif k==nl-1:
                            wu_dn[i,j] = 0.0
                            wv_dn[i,j] = 0.0
                            wu_up[i,j] = DV.Wp.values[i,j,k]*(DV.U.values[i,j,k] - DV.U.values[i,j,k-1])*dpi
                            wv_up[i,j] = DV.Wp.values[i,j,k]*(DV.V.values[i,j,k] - DV.V.values[i,j,k-1])*dpi

                            wT_dn  = 0.5*DV.Wp.values[i,j,k+1]*(PV.T.values[i,j,k]+PV.T.values[i,j,k])*dpi
                            wQT_dn = 0.5*DV.Wp.values[i,j,k+1]*(PV.QT.values[i,j,k]+ PV.QT.values[i,j,k])*dpi
                            wT_up  = 0.5*DV.Wp.values[i,j,k]*(PV.T.values[i,j,k]   + PV.T.values[i,j,k-1])*dpi
                            wQT_up = 0.5*DV.Wp.values[i,j,k]*(PV.QT.values[i,j,k]  + PV.QT.values[i,j,k-1])*dpi

                        else:
                            wu_dn[i,j] = DV.Wp.values[i,j,k+1]*(DV.U.values[i,j,k+1] - DV.U.values[i,j,k])*dpi
                            wv_dn[i,j] = DV.Wp.values[i,j,k+1]*(DV.V.values[i,j,k+1] - DV.V.values[i,j,k])*dpi
                            wu_up[i,j] = DV.Wp.values[i,j,k]*(DV.U.values[i,j,k] - DV.U.values[i,j,k-1])*dpi
                            wv_up[i,j] = DV.Wp.values[i,j,k]*(DV.V.values[i,j,k] - DV.V.values[i,j,k-1])*dpi

                            wT_dn  = 0.5*DV.Wp.values[i,j,k+1]*(PV.T.values[i,j,k+1] + PV.T.values[i,j,k])*dpi
                            wQT_dn = 0.5*DV.Wp.values[i,j,k+1]*(PV.QT.values[i,j,k+1] + PV.QT.values[i,j,k])*dpi
                            wT_up  = 0.5*DV.Wp.values[i,j,k]*(PV.T.values[i,j,k] + PV.T.values[i,j,k-1])*dpi
                            wQT_up = 0.5*DV.Wp.values[i,j,k]*(PV.QT.values[i,j,k] + PV.QT.values[i,j,k-1])*dpi

                        if k==nl-1:
                            Thermal_expension[i,j] = -DV.Wp.values[i,j,k+1]*DV.gZ.values[i,j,k]*dpi/Pr.cp
                        else:
                            Thermal_expension[i,j] = DV.Wp.values[i,j,k+1]*(DV.gZ.values[i,j,k+1]-DV.gZ.values[i,j,k])*dpi/Pr.cp

                        RHS_grid_T[i,j] = (wT_up - wT_dn - Thermal_expension[i,j]
                                            + PV.T.mp_tendency[i,j,k] + PV.T.forcing[i,j,k] + T_sur_flux[i,j])
                        RHS_grid_QT[i,j] = (wQT_up - wQT_dn
                                            + PV.QT.mp_tendency[i,j,k] + QT_sur_flux[i,j])

            Dry_Energy_laplacian = Gr.laplacian*Gr.SphericalGrid.grdtospec(Dry_Energy.base)
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vorticity.base, v_vorticity.base)
            Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(uT.base, vT.base) # Vortical_T_flux is not used
            Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(uQT.base, vQT.base) # Vortical_T_flux is not used
            Vort_forc ,Div_forc = Gr.SphericalGrid.getvrtdivspec(DV.U.forcing.base[:,:,k],DV.V.forcing.base[:,:,k])
            w_vort_up ,w_div_up = Gr.SphericalGrid.getvrtdivspec(wu_up.base, wv_up.base)
            w_vort_dn ,w_div_dn = Gr.SphericalGrid.getvrtdivspec(wu_dn.base, wv_dn.base)
            RHS_T  = Gr.SphericalGrid.grdtospec(RHS_grid_T.base)
            RHS_QT = Gr.SphericalGrid.grdtospec(RHS_grid_QT.base)

            with nogil:
                for i in range(nlm):
                    PV.Vorticity.tendency[i,k]  = (Vort_forc[i] - Divergent_momentum_flux[i]
                                                    - w_vort_up[i] - w_vort_dn[i] + Vort_sur_flux[i])
                    PV.Divergence.tendency[i,k] = (Vortical_momentum_flux[i] - Dry_Energy_laplacian[i]
                                                    - w_div_up[i] - w_div_dn[i] + Div_forc[i] + Div_sur_flux[i])

                    PV.T.tendency[i,k]  = RHS_T[i]  - Divergent_T_flux[i]
                    PV.QT.tendency[i,k] = RHS_QT[i] - Divergent_QT_flux[i]
        return
