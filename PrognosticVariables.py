import matplotlib.pyplot as plt
import scipy as sc
import netCDF4
import numpy as np
from math import *

class PrognosticVariable:
    def __init__(self, nx, ny, nl, n_spec, kind, name, units):
        self.values = np.zeros((nx,ny,nl),dtype=np.double, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.old = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.now = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.tendency = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.forcing = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        if name != 'Pressure':
            self.SurfaceFlux = np.zeros((nx,ny),dtype=np.double, order='c')
            self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.double, order='c')
            self.sp_VerticalFlux = np.zeros((n_spec,nl),dtype = np.complex, order='c')
            self.forcing = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.kind = kind
        self.name = name
        self.units = units
        return

class PrognosticVariables:
    def __init__(self, Gr):
        self.Vorticity   = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity' , 'zeta',  '1/s' )
        self.Divergence  = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Divergance', 'delta', '1/s' )
        self.T           = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'       ,  'T','K' )
        self.P           = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,  'p','pasc' )
        return

    def initialize(self, Gr, DV):
        self.Base_pressure = 100000.0
        self.T_init  = [229.0, 257.0, 295.0]
        self.P_init  = [Gr.p1, Gr.p2, Gr.p3, Gr.p_ref]

        self.Vorticity.values  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        self.Divergence.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        self.P.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers+1),  dtype=np.double, order='c'),self.P_init)
        self.T.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c'),self.T_init)
        # initilize spectral values
        for k in range(Gr.n_layers):
            self.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(self.P.values[:,:,k])
            self.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(self.T.values[:,:,k])
            self.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(self.Vorticity.values[:,:,k])
            self.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(self.Divergence.values[:,:,k])
        self.P.spectral[:,Gr.n_layers]     = Gr.SphericalGrid.grdtospec(self.P.values[:,:,Gr.n_layers])

        return

    def initialize_io(self, Stats):
        Stats.add_global_mean('global_mean_T')
        Stats.add_zonal_mean('zonal_mean_P')
        Stats.add_zonal_mean('zonal_mean_T')
        Stats.add_zonal_mean('zonal_mean_divergence')
        Stats.add_zonal_mean('zonal_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_divergence')
        Stats.add_meridional_mean('meridional_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_P')
        Stats.add_meridional_mean('meridional_mean_T')
        return

    # convert spherical data to spectral
    def physical_to_spectral(self, Gr):
        for k in range(Gr.n_layers):
            self.Vorticity.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Vorticity.values[:,:,k])
            self.Divergence.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values[:,:,k])
            self.T.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.T.values[:,:,k])
        self.P.spectral[:,Gr.n_layers] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,Gr.n_layers])
        return

    # convert spectral data to spherical
    def spectral_to_physical(self, Gr):
        for k in range(Gr.n_layers):
            self.Vorticity.values[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral[:,k])
            self.Divergence.values[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral[:,k])
            self.T.values[:,:,k]          = Gr.SphericalGrid.spectogrd(self.T.spectral[:,k])
        self.P.values[:,:,Gr.n_layers] = Gr.SphericalGrid.spectogrd(self.P.spectral[:,Gr.n_layers])
        return

    def set_old_with_now(self):
        self.Vorticity.old  = self.Vorticity.now.copy()
        self.Divergence.old = self.Divergence.now.copy()
        self.T.old          = self.T.now.copy()
        self.P.old          = self.P.now.copy()
        return

    def set_now_with_tendencies(self):
        self.Vorticity.now  = self.Vorticity.tendency.copy()
        self.Divergence.now = self.Divergence.tendency.copy()
        self.T.now          = self.T.tendency.copy()
        self.P.now          = self.P.tendency.copy()
        return

    def reset_pressures(self, Gr):
        n=Gr.n_layers
        self.P.values[:,:,0] = np.add(np.zeros_like(self.P.values[:,:,0]),self.P_init[0])
        self.P.values[:,:,1] = np.add(np.zeros_like(self.P.values[:,:,1]),self.P_init[1])
        self.P.values[:,:,2] = np.add(np.zeros_like(self.P.values[:,:,2]),self.P_init[2])
        self.P.spectral[:,0] = np.add(np.zeros_like(self.P.spectral[:,0]),self.P_init[0])
        self.P.spectral[:,1] = np.add(np.zeros_like(self.P.spectral[:,1]),self.P_init[1])
        self.P.spectral[:,2] = np.add(np.zeros_like(self.P.spectral[:,2]),self.P_init[2])
        return

    def stats_io(self, TS, Stats):
        Stats.write_global_mean('global_mean_T', self.T.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_P',self.P.values[:,:,1:4], TS.t)
        Stats.write_zonal_mean('zonal_mean_T',self.T.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_divergence',self.Divergence.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_vorticity',self.Vorticity.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_P',self.P.values[:,:,1:4], TS.t)
        Stats.write_meridional_mean('meridional_mean_T',self.T.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_divergence',self.Divergence.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_vorticity',self.Vorticity.values, TS.t)
        return

    def io(self, Gr, TS, Stats):
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Temperature',       self.T.values)
        Stats.write_2D_variable(Gr, int(TS.t),             'Pressure',          self.P.values[:,:,Gr.n_layers])
        return

    def compute_tendencies(self, Gr, PV, DV, namelist):
        ps_vrt, ps_div = Gr.SphericalGrid.getvrtdivspec(
            DV.U.values[:,:,2]*(PV.P.values[:,:,2]-PV.P.values[:,:,3]),
            DV.V.values[:,:,2]*(PV.P.values[:,:,2]-PV.P.values[:,:,3]))

        PV.P.tendency[:,3]=ps_div + DV.Wp.spectral[:,2]

        dp_ratio32sp = (PV.P.spectral[:,2]-PV.P.spectral[:,1])/(PV.P.spectral[:,3]-PV.P.spectral[:,2])

        for k in range(Gr.n_layers-1):
            u_vertical_flux = 0.5*np.multiply(DV.Wp.values[:,:,k+1],(DV.U.values[:,:,k+1]-DV.U.values[:,:,k])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))
            v_vertical_flux = 0.5*np.multiply(DV.Wp.values[:,:,k+1],(DV.V.values[:,:,k+1]-DV.V.values[:,:,k])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
            PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
            PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer

        for k in range(Gr.n_layers):
            T_high = PV.T.values[:,:,k]
            if k ==Gr.n_layers-1:
                T_low = T_high
            else:
                T_low = PV.T.values[:,:,k+1]
            PV.T.VerticalFlux[:,:,k] = 0.5*np.multiply(DV.Wp.values[:,:,k+1],
                (T_low+T_high)/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k]))

        for k in range(Gr.n_layers):
            Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
                               DV.gZ.values[:,:,k] + DV.KE.values[:,:,k])
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(
                DV.U.values[:,:,k]*(PV.Vorticity.values[:,:,k]+Gr.Coriolis),
                DV.V.values[:,:,k]*(PV.Vorticity.values[:,:,k]+Gr.Coriolis))
            Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(
                np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
                np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
            if k==0:
                vrt_flux_dn = PV.Vorticity.sp_VerticalFlux[:,k]
                vrt_flux_up = np.zeros_like(PV.Vorticity.sp_VerticalFlux[:,k])
                div_flux_dn = PV.Divergence.sp_VerticalFlux[:,k]
                div_flux_up = np.zeros_like(PV.Divergence.sp_VerticalFlux[:,k])
                T_flux_up   = np.zeros_like(PV.T.VerticalFlux[:,:,k])
                Thermal_expension = (DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
                    - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp
            elif k==Gr.n_layers-1:
                vrt_flux_dn = np.zeros_like(PV.Vorticity.sp_VerticalFlux[:,k])
                vrt_flux_up = PV.Vorticity.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                div_flux_dn = np.zeros_like(PV.Divergence.sp_VerticalFlux[:,k])
                div_flux_up = PV.Divergence.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                T_flux_up   = PV.T.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                Thermal_expension = -(DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k])
                                                          /(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp
            else:
                vrt_flux_dn = PV.Vorticity.sp_VerticalFlux[:,k]
                vrt_flux_up = PV.Vorticity.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                div_flux_dn = PV.Divergence.sp_VerticalFlux[:,k]
                div_flux_up = PV.Divergence.sp_VerticalFlux[:,k-1]*dp_ratio32sp
                T_flux_up   = PV.T.VerticalFlux[:,:,k-1]*(PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
                Thermal_expension = (DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
                    - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp

            PV.Vorticity.tendency[:,k]  = - Divergent_momentum_flux - vrt_flux_up - vrt_flux_dn + PV.Vorticity.forcing[:,k]
            PV.Divergence.tendency[:,k] =  (Vortical_momentum_flux  - Dry_Energy_laplacian
                - div_flux_up - div_flux_dn + PV.Divergence.forcing[:,k])
            PV.T.tendency[:,k] = (-Divergent_T_flux
                    +Gr.SphericalGrid.grdtospec(-Thermal_expension-PV.T.VerticalFlux[:,:,k]+T_flux_up) + PV.T.forcing[:,k])

        return
