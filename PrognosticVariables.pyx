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
import Parameters

cdef class PrognosticVariable:
    def __init__(self, nx, ny, nl, n_spec, kind, name, units):
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.old      = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.now      = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.tendency = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.float64, order='c')
        self.sp_VerticalFlux = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.forcing = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.kind = kind
        self.name = name
        self.units = units
        return

cdef class PrognosticVariables:
    def __init__(self, Pr, Gr):
        self.Vorticity   = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity'         ,'zeta' ,'1/s')
        self.Divergence  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Divergance'        ,'delta','1/s')
        self.T           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'       ,'T'    ,'K')
        self.QT          = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Specific Humidity' ,'QT'   ,'K')
        self.P           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,'p'    ,'pasc')
        return

    cpdef initialize(self, Parameters Pr):
        self.Base_pressure = 100000.0
        self.P_init        = [Pr.p1, Pr.p2, Pr.p3, Pr.p_ref]
        self.T_init        = [229.0, 257.0, 295.0]
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
    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
        for k in range(Pr.n_layers):
            self.Vorticity.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Vorticity.values.base[:,:,k])
            self.Divergence.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values.base[:,:,k])
            self.T.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.T.values.base[:,:,k])
            self.QT.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.QT.values.base[:,:,k])
        self.P.spectral.base[:,Pr.n_layers] = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,Pr.n_layers])
        return

    cpdef spectral_to_physical(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k

        for k in range(Gr.n_layers):
            self.Vorticity.values.base[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral.base[:,k])
            self.Divergence.values.base[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral.base[:,k])
            self.T.values.base[:,:,k]          = Gr.SphericalGrid.spectogrd(self.T.spectral.base[:,k])
            self.QT.values.base[:,:,k]         = Gr.SphericalGrid.spectogrd(self.QT.spectral.base[:,k])
        self.P.values.base[:,:,Pr.n_layers] = Gr.SphericalGrid.spectogrd(np.copy(self.P.spectral.base[:,Pr.n_layers]))
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
        for k in range(Pr.n_layers):
            self.P.values.base[:,:,k] = np.add(np.zeros_like(self.P.values.base[:,:,k]),self.P_init[k])
        return
    # this should be done in time intervals and save each time new files,not part of stats 

    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_T', self.T.values, TS.t)
        Stats.write_global_mean('global_mean_QT', self.QT.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_P',self.P.values[:,:,1:4], TS.t)
        Stats.write_zonal_mean('zonal_mean_T',self.T.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_QT',self.QT.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_divergence',self.Divergence.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_vorticity',self.Vorticity.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_P',self.P.values[:,:,1:4], TS.t)
        Stats.write_meridional_mean('meridional_mean_T',self.T.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_QT',self.QT.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_divergence',self.Divergence.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_vorticity',self.Vorticity.values, TS.t)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, 'Temperature',       self.T.values)
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Pr, int(TS.t),1,           'Pressure',          self.P.values[:,:,Pr.n_layers])
        return

    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        cdef:
            Py_ssize_t k
            double [:] ps_vrt
            double [:] ps_div
            double [:] Vortical_P_flux
            double [:] Divergent_P_flux

        nz = Pr.n_layers

        ps_vrt, ps_div = Gr.SphericalGrid.getvrtdivspec(
            DV.U.values[:,:,2]*(PV.P.values[:,:,2]-PV.P.values[:,:,3]),
            DV.V.values[:,:,2]*(PV.P.values[:,:,2]-PV.P.values[:,:,3]))
        # compute surface pressure tendency
        Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(
                         np.multiply(DV.U.values[:,:,nz-1],np.subtract(PV.P.values[:,:,nz-1],PV.P.values[:,:,nz])),
                         np.multiply(DV.V.values[:,:,nz-1],np.subtract(PV.P.values[:,:,nz-1],PV.P.values[:,:,nz]))) # Vortical_P_flux is not used
        PV.P.tendency.base[:,nz] = Divergent_P_flux + DV.Wp.spectral.base[:,nz-1]

    cpdef compute_tendencies(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        cdef:
            Py_ssize_t k
            double [:] ps_vrt
            double [:] ps_div
            double [:] Vortical_P_flux
            double [:] Divergent_P_flux
            double [:] dp_ratio32sp
            double [:] u_vertical_flux
            double [:] v_vertical_flux
            double [:] Vortical_momentum_flux
            double [:] Divergent_momentum_flux
            double [:,:] T_high
            double [:,:] QT_high
            double [:,:] T_low
            double [:,:] QT_low
            double [:,:] Dry_Energy_laplacian
            double [:,:] Vortical_T_flux
            double [:,:] Divergent_T_flux
            double [:,:] Vortical_QT_flux
            double [:,:] Divergent_QT_flux
            double [:,:] vrt_flux_dn
            double [:,:] vrt_flux_up
            double [:,:] div_flux_dn
            double [:,:] div_flux_up
            double [:,:] T_flux_up
            double [:,:] QT_flux_up
            double [:,:] Thermal_expension



        ps_vrt, ps_div = Gr.SphericalGrid.getvrtdivspec(
            np.multiply(DV.U.values[:,:,2],np.subtract(PV.P.values[:,:,2],PV.P.values[:,:,3])),
            np.multiply(DV.V.values[:,:,2],np.subtract(PV.P.values[:,:,2],PV.P.values[:,:,3])))

        PV.P.tendency.base[:,3]= np.add(ps_div, DV.Wp.spectral[:,2])

        dp_ratio32sp = np.divide(np.subtract(PV.P.spectral[:,2],PV.P.spectral[:,1]),
                                 np.subtract(PV.P.spectral[:,3],PV.P.spectral[:,2]))

        for k in range(Pr.n_layers-1):
            u_vertical_flux = 0.5*np.multiply(DV.Wp.values[:,:,k+1],(DV.U.values[:,:,k+1]-DV.U.values[:,:,k])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))
            v_vertical_flux = 0.5*np.multiply(DV.Wp.values[:,:,k+1],(DV.V.values[:,:,k+1]-DV.V.values[:,:,k])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
            PV.Vorticity.sp_VerticalFlux.base[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
            PV.Divergence.sp_VerticalFlux.base[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer

        for k in range(Pr.n_layers):
            T_high = PV.T.values[:,:,k]
            QT_high = PV.QT.values[:,:,k]
            if k ==Pr.n_layers-1:
                T_low = T_high
                QT_low = QT_high
            else:
                T_low = PV.T.values[:,:,k+1]
                QT_low = PV.QT.values[:,:,k+1]
            PV.T.VerticalFlux.base[:,:,k] = 0.5*np.divide(np.multiply(DV.Wp.values[:,:,k+1],
                (T_low+T_high),np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])))
            PV.QT.VerticalFlux.base[:,:,k] = 0.5*np.divide(np.multiply(DV.Wp.values[:,:,k+1],
                (QT_low+QT_high),np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])))

        for k in range(Pr.n_layers):
            Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
                               DV.gZ.values[:,:,k] + DV.KE.values[:,:,k])
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(
                DV.U.values[:,:,k]*(PV.Vorticity.values[:,:,k]+Gr.Coriolis),
                DV.V.values[:,:,k]*(PV.Vorticity.values[:,:,k]+Gr.Coriolis))
            Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(
                np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
                np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
            Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(
                np.multiply(DV.U.values[:,:,k],PV.QT.values[:,:,k]),
                np.multiply(DV.V.values[:,:,k],PV.QT.values[:,:,k])) # Vortical_T_flux is not used
            if k==0:
                vrt_flux_dn = PV.Vorticity.sp_VerticalFlux[:,k]
                vrt_flux_up = np.zeros_like(PV.Vorticity.sp_VerticalFlux[:,k])
                div_flux_dn = PV.Divergence.sp_VerticalFlux[:,k]
                div_flux_up = np.zeros_like(PV.Divergence.sp_VerticalFlux[:,k])
                T_flux_up   = np.zeros_like(PV.T.VerticalFlux[:,:,k])
                QT_flux_up   = np.zeros_like(PV.QT.VerticalFlux[:,:,k])
                Thermal_expension = (DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
                    - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Pr.cp
            elif k==Pr.n_layers-1:
                vrt_flux_dn = np.zeros_like(PV.Vorticity.sp_VerticalFlux[:,k])
                vrt_flux_up = PV.Vorticity.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                div_flux_dn = np.zeros_like(PV.Divergence.sp_VerticalFlux[:,k])
                div_flux_up = PV.Divergence.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                # check if you can use the dpratio here        
                T_flux_up   = PV.T.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                QT_flux_up  = PV.QT.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                Thermal_expension = -np.divide(np.divide(np.multiply(DV.Wp.values[:,:,k+1],DV.gZ.values[:,:,k])
                                                          ,np.subtract(PV.P.values[:,:,k+1], PV.P.values[:,:,k])),Pr.cp)

            else:
                vrt_flux_dn = PV.Vorticity.sp_VerticalFlux[:,k]
                vrt_flux_up = np.multiply(PV.Vorticity.sp_VerticalFlux[:,k-1],dp_ratio32sp)
                div_flux_dn = PV.Divergence.sp_VerticalFlux[:,k]
                div_flux_up = np.multiply(PV.Divergence.sp_VerticalFlux[:,k-1],dp_ratio32sp)
                T_flux_up   = PV.T.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                QT_flux_up  = PV.QT.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                Thermal_expension = (DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
                    - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Pr.cp

            PV.Vorticity.tendency.base[:,k]  = - Divergent_momentum_flux - vrt_flux_up - vrt_flux_dn + PV.Vorticity.forcing[:,k]
            PV.Divergence.tendency.base[:,k] =  (Vortical_momentum_flux  - Dry_Energy_laplacian
                - div_flux_up - div_flux_dn + PV.Divergence.forcing[:,k])
            PV.T.tendency.base[:,k] = (-Divergent_T_flux
                    +Gr.SphericalGrid.grdtospec(-Thermal_expension-PV.T.VerticalFlux[:,:,k]+T_flux_up + MP.dTdt[:,:,k] ) + PV.T.forcing[:,k])
            PV.QT.tendency.base[:,k] = (-Divergent_QT_flux
                    +Gr.SphericalGrid.grdtospec(-Thermal_expension-PV.QT.VerticalFlux[:,:,k]+QT_flux_up + MP.dQTdt[:,:,k]) + PV.QT.forcing[:,k])
        return