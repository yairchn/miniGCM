import matplotlib.pyplot as plt
import scipy as sc
import netCDF4
import numpy as np
from math import *

class PrognosticVariable:
    def __init__(self, nx, ny, nl, n_spec, kind, name, units):
        self.values   = np.zeros((nx,ny,nl), dtype = np.double,  order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.old      = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.now      = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.tendency = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.forcing  = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        if name != 'Pressure':
            self.SurfaceFlux     = np.zeros((nx,ny),     dtype = np.double,  order='c')
            self.VerticalFlux    = np.zeros((nx,ny,nl+1),dtype = np.double,  order='c')
            self.sp_VerticalFlux = np.zeros((n_spec,nl), dtype = np.complex, order='c')
            self.forcing         = np.zeros((n_spec,nl), dtype = np.complex, order='c')
        self.kind = kind
        self.name = name
        self.units = units
        return

class PrognosticVariables:
    def __init__(self, Pr, Gr):
        self.Vorticity   = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity'         ,'zeta' ,'1/s')
        self.Divergence  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Divergance'        ,'delta','1/s')
        self.T           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'       ,'T'    ,'K')
        self.QT          = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Specific Humidity' ,'QT'   ,'K')
        self.P           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,'p'    ,'pasc')
        return

    def initialize(self, Pr):
        self.Base_pressure = 100000.0
        self.P_init        = [Pr.p1, Pr.p2, Pr.p3, Pr.p_ref]
        self.T_init        = [229.0, 257.0, 295.0]
        return

    def initialize_io(self, Stats):
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
    def physical_to_spectral(self, Pr, Gr):
        for k in range(Pr.n_layers):
            self.Vorticity.spectral[:,k]  = Gr.SphericalGrid.grdtospec(self.Vorticity.values[:,:,k])
            self.Divergence.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values[:,:,k])
            self.T.spectral[:,k]          = Gr.SphericalGrid.grdtospec(self.T.values[:,:,k])
        self.P.spectral[:,Pr.n_layers]    = Gr.SphericalGrid.grdtospec(self.P.values[:,:,Pr.n_layers])
        return

    # convert spectral data to spherical
    def spectral_to_physical(self, Pr, Gr):
        for k in range(Pr.n_layers):
            self.Vorticity.values[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral[:,k])
            self.Divergence.values[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral[:,k])
            self.T.values[:,:,k]          = Gr.SphericalGrid.spectogrd(self.T.spectral[:,k])
            self.QT.values[:,:,k]         = Gr.SphericalGrid.spectogrd(self.QT.spectral[:,k])
        self.P.values[:,:,Pr.n_layers] = Gr.SphericalGrid.spectogrd(self.P.spectral[:,Pr.n_layers])
        return

    def set_old_with_now(self):
        self.Vorticity.old  = self.Vorticity.now.copy()
        self.Divergence.old = self.Divergence.now.copy()
        self.T.old          = self.T.now.copy()
        self.QT.old         = self.QT.now.copy()
        self.P.old          = self.P.now.copy()
        return

    def set_now_with_tendencies(self):
        self.Vorticity.now  = self.Vorticity.tendency.copy()
        self.Divergence.now = self.Divergence.tendency.copy()
        self.T.now          = self.T.tendency.copy()
        self.QT.now         = self.QT.tendency.copy()
        self.P.now          = self.P.tendency.copy()
        return

    def reset_pressures(self, Pr, Gr):
        for k in range(Pr.n_layers):
            self.P.values[:,:,k] = np.add(np.zeros_like(self.P.values[:,:,k]),self.P_init[k])
            self.P.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,k])
        return

    def stats_io(self, TS, Stats):
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

    def io(self, Pr, TS, Stats):
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, self.Vorticity.name,   self.Vorticity.values)
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, self.Divergence.name,  self.Divergence.values)
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, self.T.name,           self.T.values)
        Stats.write_3D_variable(Pr, int(TS.t),Pr.n_layers, self.QT.name,          self.QT.values)
        Stats.write_2D_variable(Pr, int(TS.t),             self.P.name,           self.P.values[:,:,Pr.n_layers])
        return

    def compute_tendencies(self, Pr, Gr, PV, DV, MP):
        ps_vrt, ps_div = Gr.SphericalGrid.getvrtdivspec(
            DV.U.values[:,:,2]*(PV.P.values[:,:,2]-PV.P.values[:,:,3]),
            DV.V.values[:,:,2]*(PV.P.values[:,:,2]-PV.P.values[:,:,3]))

        PV.P.tendency[:,3]=ps_div + DV.Wp.spectral[:,2]

        dp_ratio32sp = (PV.P.spectral[:,2]-PV.P.spectral[:,1])/(PV.P.spectral[:,3]-PV.P.spectral[:,2])

        for k in range(Pr.n_layers-1):
            u_vertical_flux = 0.5*np.multiply(DV.Wp.values[:,:,k+1],(DV.U.values[:,:,k+1]-DV.U.values[:,:,k])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))
            v_vertical_flux = 0.5*np.multiply(DV.Wp.values[:,:,k+1],(DV.V.values[:,:,k+1]-DV.V.values[:,:,k])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
            PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
            PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer

        for k in range(Pr.n_layers):
            T_high = PV.T.values[:,:,k]
            QT_high = PV.QT.values[:,:,k]
            if k ==Pr.n_layers-1:
                T_low = T_high
                QT_low = QT_high
            else:
                T_low = PV.T.values[:,:,k+1]
                QT_low = PV.QT.values[:,:,k+1]
            PV.T.VerticalFlux[:,:,k] = 0.5*np.multiply(DV.Wp.values[:,:,k+1],
                (T_low+T_high)/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))
            PV.QT.VerticalFlux[:,:,k] = 0.5*np.multiply(DV.Wp.values[:,:,k+1],
                (QT_low+QT_high)/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))

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
                T_flux_up   = PV.T.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                QT_flux_up  = PV.QT.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                Thermal_expension = -(DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k])
                                                          /(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Pr.cp
            else:
                vrt_flux_dn = PV.Vorticity.sp_VerticalFlux[:,k]
                vrt_flux_up = PV.Vorticity.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                div_flux_dn = PV.Divergence.sp_VerticalFlux[:,k]
                div_flux_up = PV.Divergence.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                T_flux_up   = PV.T.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                QT_flux_up  = PV.QT.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                Thermal_expension = (DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
                    - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Pr.cp

            PV.Vorticity.tendency[:,k]  = - Divergent_momentum_flux - vrt_flux_up - vrt_flux_dn + PV.Vorticity.forcing[:,k]
            PV.Divergence.tendency[:,k] =  (Vortical_momentum_flux  - Dry_Energy_laplacian
                - div_flux_up - div_flux_dn + PV.Divergence.forcing[:,k])
            PV.T.tendency[:,k] = (-Divergent_T_flux
                    +Gr.SphericalGrid.grdtospec(-Thermal_expension-PV.T.VerticalFlux[:,:,k]+T_flux_up+ MP.dTdt[:,:,k]) + PV.T.forcing[:,k]) # 
            PV.QT.tendency[:,k] = (-Divergent_QT_flux
                    +Gr.SphericalGrid.grdtospec(-Thermal_expension-PV.QT.VerticalFlux[:,:,k]+QT_flux_up+ MP.dQTdt[:,:,k]) + PV.QT.forcing[:,k]) # 

        return
