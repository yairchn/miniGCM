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

    cpdef reset_pressures(self, Parameters Pr):
        cdef:
            Py_ssize_t nl = Pr.n_layers

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
            double T_high, QT_high, T_low, QT_low
            double complex [:] Vortical_P_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_P_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vortical_momentum_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_momentum_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vortical_T_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_T_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vortical_QT_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Divergent_QT_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Dry_Energy_laplacian = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] vrt_flux_dn = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] vrt_flux_up = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] div_flux_dn = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] div_flux_up = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vort_forc = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Div_forc = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Vort_sur_flux = np.zeros((nlm), dtype = np.complex, order='c')
            double complex [:] Div_sur_flux  = np.zeros((nlm), dtype = np.complex, order='c')
            double [:,:] u_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] v_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] Dry_Energy = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] u_vorticity_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] v_vorticity_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] u_T_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] v_T_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] u_QT_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] v_QT_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] T_flux_up = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] QT_flux_up = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] Thermal_expension = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] RHS_grid_T = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] RHS_grid_QT = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] T_sur_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:] QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order='c')
            double [:,:,:] dp = np.zeros((nx, ny, nl+1),dtype=np.float64, order='c')

        Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(
            np.multiply(DV.U.values[:,:,2],np.subtract(PV.P.values[:,:,2],PV.P.values[:,:,3])),
            np.multiply(DV.V.values[:,:,2],np.subtract(PV.P.values[:,:,2],PV.P.values[:,:,3])))

        PV.P.tendency.base[:,3] = np.add(Divergent_P_flux, DV.Wp.spectral[:,2])
        dp_ratio32sp = np.divide(np.subtract(PV.P.spectral[:,2],PV.P.spectral[:,1]), np.subtract(PV.P.spectral[:,3],PV.P.spectral[:,2]))

        with nogil:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nl):
                        dp[i,j,k] = PV.P.values[i,j,k+1]-PV.P.values[i,j,k]

        for k in range(nl-1):
            with nogil:
                for i in range(nx):
                    for j in range(ny):
                        u_vertical_flux[i,j] = 0.5*(DV.Wp.values[i,j,k+1]*((DV.U.values[i,j,k+1]-DV.U.values[i,j,k])/dp[i,j,k]))
                        v_vertical_flux[i,j] = 0.5*(DV.Wp.values[i,j,k+1]*((DV.V.values[i,j,k+1]-DV.V.values[i,j,k])/dp[i,j,k]))
        for k in range(nl-1):
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux.base, v_vertical_flux.base)
            PV.Vorticity.sp_VerticalFlux.base[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
            PV.Divergence.sp_VerticalFlux.base[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer


        with nogil:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nl):
                        T_high = PV.T.values[i,j,k]
                        QT_high = PV.QT.values[i,j,k]
                        if k ==nl-1:
                            T_low = T_high
                            QT_low = QT_high
                        else:
                            T_low = PV.T.values[i,j,k+1]
                            QT_low = PV.QT.values[i,j,k+1]
                        PV.T.VerticalFlux[i,j,k]  = 0.5*DV.Wp.values[i,j,k+1]*(T_low +T_high) /dp[i,j,k]
                        PV.QT.VerticalFlux[i,j,k] = 0.5*DV.Wp.values[i,j,k+1]*(QT_low+QT_high)/dp[i,j,k]


        for k in range(nl):
            with nogil:
                for i in range(nx):
                    for j in range(ny):
                        Dry_Energy[i,j]       = DV.gZ.values[i,j,k]+DV.KE.values[i,j,k]
                        u_vorticity_flux[i,j] = DV.U.values[i,j,k] * (PV.Vorticity.values[i,j,k]+Gr.Coriolis[i,j])
                        v_vorticity_flux[i,j] = DV.V.values[i,j,k] * (PV.Vorticity.values[i,j,k]+Gr.Coriolis[i,j])
                        u_T_flux[i,j]         = DV.U.values[i,j,k] * PV.T.values[i,j,k]
                        v_T_flux[i,j]         = DV.V.values[i,j,k] * PV.T.values[i,j,k]
                        u_QT_flux[i,j]        = DV.U.values[i,j,k] * PV.QT.values[i,j,k]
                        v_QT_flux[i,j]        = DV.V.values[i,j,k] * PV.QT.values[i,j,k]

            Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(Dry_Energy.base)
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vorticity_flux.base,v_vorticity_flux.base)
            Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(u_T_flux.base,v_T_flux.base) # Vortical_T_flux is not used
            Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(u_QT_flux.base,v_QT_flux.base) # Vortical_T_flux is not used


        for k in range(nl):
            with nogil:
                for i in range(nlm):
                    if k==0:
                        vrt_flux_dn[i] = PV.Vorticity.sp_VerticalFlux[i,k]
                        div_flux_dn[i] = PV.Divergence.sp_VerticalFlux[i,k]
                        vrt_flux_up[i] = 0.0
                        div_flux_up[i] = 0.0
                    elif k==nl-1:
                        vrt_flux_dn[i] = 0.0
                        div_flux_dn[i] = 0.0
                        vrt_flux_up[i] = (PV.Vorticity.sp_VerticalFlux[i,k-1]*
                            ((PV.P.spectral[i,2]-PV.P.spectral[i,1])/(PV.P.spectral[i,3]-PV.P.spectral[i,2])))
                        div_flux_up[i] = (PV.Divergence.sp_VerticalFlux[i,k-1]*
                            ((PV.P.spectral[i,2]-PV.P.spectral[i,1])/(PV.P.spectral[i,3]-PV.P.spectral[i,2])))
                    else:
                        vrt_flux_dn[i] = 0.0
                        div_flux_dn[i] = 0.0
                        vrt_flux_up[i] = (PV.Vorticity.sp_VerticalFlux[i,k-1]*
                            ((PV.P.spectral[i,2]-PV.P.spectral[i,1])/(PV.P.spectral[i,3]-PV.P.spectral[i,2])))
                        div_flux_up[i] = (PV.Divergence.sp_VerticalFlux[i,k-1]*
                            ((PV.P.spectral[i,2]-PV.P.spectral[i,1])/(PV.P.spectral[i,3]-PV.P.spectral[i,2])))

                for i in range(nx):
                    for j in range(ny):
                        if k==0:
                            T_flux_up[i,j]   = 0.0
                            QT_flux_up[i,j]  = 0.0
                            Thermal_expension[i,j] = DV.Wp.values[i,j,k+1]*(DV.gZ.values[i,j,k+1]-DV.gZ.values[i,j,k])/(dp[i,j,k]*Pr.cp)
                        elif k==nl-1:
                            T_flux_up[i,j]   = PV.T.VerticalFlux[i,j,k-1] *(PV.P.values[i,j,k]-PV.P.values[i,j,k-1])/dp[i,j,k]
                            QT_flux_up[i,j]  = PV.QT.VerticalFlux[i,j,k-1]*(PV.P.values[i,j,k]-PV.P.values[i,j,k-1])/dp[i,j,k]
                            Thermal_expension[i,j] = -DV.Wp.values[i,j,k+1]*DV.gZ.values[i,j,k]/(dp[i,j,k]*Pr.cp)
                        else:
                            T_flux_up[i,j]  = PV.T.VerticalFlux[i,j,k-1] *dp[i,j,k-1]/dp[i,j,k]
                            QT_flux_up[i,j] = PV.QT.VerticalFlux[i,j,k-1]*dp[i,j,k-1]/dp[i,j,k]
                            Thermal_expension[i,j] = DV.Wp.values[i,j,k+1]*(DV.gZ.values[i,j,k+1]-DV.gZ.values[i,j,k])/(dp[i,j,k]*Pr.cp)

            if k<nl-1:
                Vort_sur_flux = np.zeros((nlm),dtype = np.complex, order='c')
                Div_sur_flux  = np.zeros((nlm),dtype = np.complex, order='c')
                T_sur_flux    = np.zeros((nx, ny),dtype=np.float64, order='c')
                QT_sur_flux   = np.zeros((nx, ny),dtype=np.float64, order='c')
            else:
                Vort_sur_flux ,Div_sur_flux = Gr.SphericalGrid.getvrtdivspec(DV.U.SurfaceFlux.base, DV.V.SurfaceFlux.base)
                T_sur_flux = PV.T.SurfaceFlux
                QT_sur_flux = PV.QT.SurfaceFlux

            Vort_forc ,Div_forc = Gr.SphericalGrid.getvrtdivspec(DV.U.forcing.base[:,:,k],DV.V.forcing.base[:,:,k])

            with nogil:
                for i in range(nlm):
                    PV.Vorticity.tendency[i,k]  = (Vort_forc[i] - Divergent_momentum_flux[i]
                                             - vrt_flux_up[i] - vrt_flux_dn[i] + Vort_sur_flux[i])
                    PV.Divergence.tendency[i,k] = (Vortical_momentum_flux[i] - Dry_Energy_laplacian[i]
                                      - div_flux_up[i] - div_flux_dn[i] + Div_forc[i] + Div_sur_flux[i])
                for i in range(nx):
                    for j in range(ny):
                        RHS_grid_T[i,j] = (T_flux_up[i,j]+PV.T.mp_tendency[i,j,k]-Thermal_expension[i,j]
                                   -PV.T.VerticalFlux[i,j,k] + PV.T.forcing[i,j,k] + T_sur_flux[i,j])
                        RHS_grid_QT[i,j] = (QT_flux_up[i,j] + PV.QT.mp_tendency[i,j,k]
                                          - PV.QT.VerticalFlux[i,j,k] + QT_sur_flux[i,j])

            PV.T.tendency.base[:,k]  = np.subtract(Gr.SphericalGrid.grdtospec(RHS_grid_T.base), Divergent_T_flux)
            PV.QT.tendency.base[:,k] = np.subtract(Gr.SphericalGrid.grdtospec(RHS_grid_QT.base),Divergent_QT_flux)

        return
