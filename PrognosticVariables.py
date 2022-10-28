import numpy as np
from DiagnosticVariables import DiagnosticVariables
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
import Microphysics
from NetCDFIO import NetCDFIO_Stats
from Parameters import Parameters
from TimeStepping import TimeStepping

class PrognosticVariable:
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

class PrognosticVariables:
    def __init__(self, Pr, Gr, namelist):
        self.Vorticity   = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity'         ,'zeta' ,'1/s')
        self.Divergence  = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Divergance'        ,'delta','1/s')
        self.T           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'       ,'T'    ,'K')
        self.QT          = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,  Gr.SphericalGrid.nlm,'Specific Humidity' ,'QT'   ,'K')
        self.P           = PrognosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,'p'    ,'pasc')
        return

    def initialize(self, Pr):
        self.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        return

    def initialize_io(self, Pr, Stats):
        Stats.add_global_mean('global_mean_T')
        Stats.add_zonal_mean('zonal_mean_T')
        Stats.add_zonal_mean('zonal_mean_divergence')
        Stats.add_zonal_mean('zonal_mean_vorticity')
        Stats.add_surface_zonal_mean('zonal_mean_Ps')
        Stats.add_surface_zonal_mean('T_SurfaceFlux')
        Stats.add_surface_zonal_mean('QT_SurfaceFlux')
        Stats.add_meridional_mean('meridional_mean_divergence')
        Stats.add_meridional_mean('meridional_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_T')
        Stats.add_surface_meridional_mean('meridional_mean_Ps')
        if Pr.moist_index > 0.0:
            Stats.add_zonal_mean('zonal_mean_dTdt')
            Stats.add_meridional_mean('meridional_mean_dTdt')
            Stats.add_global_mean('global_mean_dTdt')
            Stats.add_global_mean('global_mean_QT')
            Stats.add_zonal_mean('zonal_mean_QT')
            Stats.add_meridional_mean('meridional_mean_QT')
        return

    # convert spherical data to spectral
    # I needto define this function to ast on a general variable
    def physical_to_spectral(self, Pr, Gr):
        nl = Pr.n_layers
        self.P.spectral[:,nl] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,nl])
        for k in range(nl):
            self.Vorticity.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Vorticity.values[:,:,k])
            self.Divergence.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values[:,:,k])
            self.T.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.T.values[:,:,k])
            if Pr.moist_index > 0.0:
                self.QT.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.QT.values[:,:,k])
        return

    def spectral_to_physical(self, Pr, Gr):
        nl = Pr.n_layers

        self.P.values[:,:,nl] = Gr.SphericalGrid.spectogrd(np.copy(self.P.spectral[:,nl]))
        for k in range(nl):
            self.Vorticity.values[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral[:,k])
            self.Divergence.values[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral[:,k])
            self.T.values[:,:,k]          = Gr.SphericalGrid.spectogrd(self.T.spectral[:,k])
            if Pr.moist_index > 0.0:
                self.QT.values[:,:,k]         = np.clip(Gr.SphericalGrid.spectogrd(self.QT.spectral[:,k]), 0.0, 1.0)
        return

    # quick utility to set arrays with values in the "new" arrays
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

    def reset_pressures_and_bcs(self, Pr, DV):
        nl = Pr.n_layers

        DV.gZ.values[:,:,nl] = np.zeros_like(self.P.values[:,:,nl])
        DV.Wp.values[:,:,0]  = np.zeros_like(self.P.values[:,:,nl])
        for k in range(nl):
            self.P.values[:,:,k] = np.add(np.zeros_like(self.P.values[:,:,k]),Pr.pressure_levels[k])
            self.P.spectral[:,k] = np.add(np.zeros_like(self.P.spectral[:,k]),Pr.pressure_levels[k])
        return
    # this should be done in time intervals and save each time new files,not part of stats

    def stats_io(self, Gr, Pr, Stats):
        nl = Pr.n_layers

        Stats.write_global_mean(Gr, 'global_mean_T', self.T.values)
        Stats.write_surface_zonal_mean('zonal_mean_Ps',self.P.values[:,:,nl])
        Stats.write_surface_zonal_mean('T_SurfaceFlux', self.T.SurfaceFlux)
        Stats.write_surface_zonal_mean('QT_SurfaceFlux',self.QT.SurfaceFlux)
        Stats.write_zonal_mean('zonal_mean_T',self.T.values)
        Stats.write_zonal_mean('zonal_mean_divergence',self.Divergence.values)
        Stats.write_zonal_mean('zonal_mean_vorticity',self.Vorticity.values)
        # Stats.write_surface_meridional_mean('meridional_mean_Ps',self.P.values[:,:,-1])
        Stats.write_meridional_mean('meridional_mean_T',self.T.values)
        Stats.write_meridional_mean('meridional_mean_divergence',self.Divergence.values)
        Stats.write_meridional_mean('meridional_mean_vorticity',self.Vorticity.values)
        if Pr.moist_index > 0.0:
            Stats.write_zonal_mean('zonal_mean_dTdt',self.T.mp_tendency)
            Stats.write_meridional_mean('meridional_mean_dTdt',self.T.mp_tendency)
            Stats.write_global_mean(Gr, 'global_mean_dTdt', self.T.mp_tendency)
            Stats.write_global_mean(Gr, 'global_mean_QT', self.QT.values)
            Stats.write_zonal_mean('zonal_mean_QT',self.QT.values)
            Stats.write_meridional_mean('meridional_mean_QT',self.QT.values)
        return

    def io(self, Pr, Gr, TS, Stats):
        nl = Pr.n_layers
        nlm = Gr.SphericalGrid.nlm

        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'Temperature',       self.T.values)
        Stats.write_2D_variable(Pr, int(TS.t),    'Pressure',          self.P.values[:,:,nl])
        Stats.write_2D_variable(Pr, int(TS.t),    'T_SurfaceFlux',     self.T.SurfaceFlux)
        # Stats.write_spectral_field(Pr, int(TS.t),nlm, nl, 'Vorticity_forcing', self.Vorticity.sp_forcing)
        if Pr.moist_index > 0.0:
            Stats.write_3D_variable(Pr, int(TS.t),nl, 'Specific_humidity', self.QT.values)
            Stats.write_3D_variable(Pr, int(TS.t),nl, 'dQTdt',             self.QT.mp_tendency[:,:,0:nl])
            Stats.write_2D_variable(Pr, int(TS.t),    'QT_SurfaceFlux',    self.QT.SurfaceFlux)
        return

    def compute_tendencies(self, Pr, Gr, PV, DV):
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers
        nlm = Gr.SphericalGrid.nlm

        w_vort_up = np.zeros((nlm), dtype = np.complex, order ='c')
        w_vort_dn = np.zeros((nlm), dtype = np.complex, order ='c')
        w_div_up  = np.zeros((nlm), dtype = np.complex, order ='c')
        w_div_dn  = np.zeros((nlm), dtype = np.complex, order ='c')
        Vort_forc = np.zeros((nlm), dtype = np.complex, order ='c')
        Div_forc  = np.zeros((nlm), dtype = np.complex, order ='c')
        RHS_T     = np.zeros((nlm), dtype = np.complex, order ='c')
        RHS_QT    = np.zeros((nlm), dtype = np.complex, order ='c')
        Vort_sur_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Div_sur_flux  = np.zeros((nlm), dtype = np.complex, order ='c')
        Vortical_T_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Divergent_T_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Vortical_QT_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Vortical_P_flux  = np.zeros((nlm), dtype = np.complex, order ='c')
        Divergent_P_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Divergent_QT_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Dry_Energy_laplacian = np.zeros((nlm), dtype = np.complex, order ='c')
        Vortical_momentum_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Divergent_momentum_flux = np.zeros((nlm), dtype = np.complex, order ='c')
        Wp3_spectral = np.zeros((nlm), dtype = np.complex, order ='c')

        uT  = np.zeros((nx, ny),dtype=np.float64, order ='c')
        vT  = np.zeros((nx, ny),dtype=np.float64, order ='c')
        uQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
        vQT = np.zeros((nx, ny),dtype=np.float64, order ='c')
        wu_dn = np.zeros((nx, ny),dtype=np.float64, order ='c')
        wv_dn = np.zeros((nx, ny),dtype=np.float64, order ='c')
        wu_up = np.zeros((nx, ny),dtype=np.float64, order ='c')
        wv_up = np.zeros((nx, ny),dtype=np.float64, order ='c')
        dwTdp_up = np.zeros((nx, ny),dtype=np.float64, order ='c')
        dwTdp_dn = np.zeros((nx, ny),dtype=np.float64, order ='c')
        T_turbfluxdiv = np.zeros((nx, ny),dtype=np.float64, order ='c')
        QT_turbfluxdiv = np.zeros((nx, ny),dtype=np.float64, order ='c')
        Dry_Energy = np.zeros((nx, ny),dtype=np.float64, order ='c')
        u_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
        v_vorticity = np.zeros((nx, ny),dtype=np.float64, order ='c')
        RHS_grid_T  = np.zeros((nx, ny),dtype=np.float64, order ='c')
        RHS_grid_QT = np.zeros((nx, ny),dtype=np.float64, order ='c')
        T_sur_flux  = np.zeros((nx, ny),dtype=np.float64, order ='c')
        QT_sur_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
        u_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
        v_vertical_flux = np.zeros((nx, ny),dtype=np.float64, order ='c')
        dpi = np.zeros_like(PV.P.values[:,:,0])


        Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(
            np.multiply(DV.U.values[:,:,nl-1],np.subtract(PV.P.values[:,:,nl-1],PV.P.values[:,:,nl])),
            np.multiply(DV.V.values[:,:,nl-1],np.subtract(PV.P.values[:,:,nl-1],PV.P.values[:,:,nl])))

        Wp3_spectral = Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,nl-1])
        dpsdx, dpsdy = Gr.SphericalGrid.getgrad(PV.P.spectral[:,nl])
        PV.P.tendency[:,nl] = np.add(Divergent_P_flux, Wp3_spectral)

        for k in range(nl):
            if k==nl-1:
                Vort_sur_flux ,Div_sur_flux = Gr.SphericalGrid.getvrtdivspec(DV.U.SurfaceFlux, DV.V.SurfaceFlux)
                T_turb_flux  = PV.T.SurfaceFlux
                QT_turb_flux = PV.QT.SurfaceFlux

            dpi = np.divide(1.0,(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))
            Dry_Energy = np.add(np.divide(np.add(DV.gZ.values[:,:,k+1],DV.gZ.values[:,:,k]),2.0), DV.KE.values[:,:,k])
            u_vorticity = np.multiply(DV.U.values[:,:,k], np.add(PV.Vorticity.values[:,:,k],Gr.Coriolis[:,:]))
            v_vorticity = np.multiply(DV.V.values[:,:,k], np.add(PV.Vorticity.values[:,:,k],Gr.Coriolis[:,:]))
            if nl-1>0:
                if k==0:
                    wu_dn = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k+1],np.subtract(DV.U.values[:,:,k+1], DV.U.values[:,:,k])),dpi))
                    wv_dn = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k+1],np.subtract(DV.V.values[:,:,k+1], DV.V.values[:,:,k])),dpi))
                    wu_up = np.zeros_like(wu_up)
                    wv_up = np.zeros_like(wv_up)
                elif k==nl-1:
                    wu_up = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k],np.subtract(DV.U.values[:,:,k], DV.U.values[:,:,k-1])),dpi))
                    wv_up = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k],np.subtract(DV.V.values[:,:,k], DV.V.values[:,:,k-1])),dpi))
                    wu_dn = np.zeros_like(wu_dn)
                    wv_dn = np.zeros_like(wv_dn)
                else:
                    wu_up = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k],np.subtract(DV.U.values[:,:,k], DV.U.values[:,:,k-1])),dpi))
                    wv_up = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k],np.subtract(DV.V.values[:,:,k], DV.V.values[:,:,k-1])),dpi))
                    wu_dn = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k+1],np.subtract(DV.U.values[:,:,k+1], DV.U.values[:,:,k])),dpi))
                    wv_dn = np.multiply(0.5,np.multiply(np.multiply(DV.Wp.values[:,:,k+1],np.subtract(DV.V.values[:,:,k+1], DV.V.values[:,:,k])),dpi))

            uT = np.multiply(DV.U.values[:,:,k], PV.T.values[:,:,k])
            vT = np.multiply(DV.V.values[:,:,k], PV.T.values[:,:,k])
            if nl-1>0:
                if k==0:
                    dwTdp_dn = -0.5*DV.Wp.values[:,:,k+1]*(PV.T.values[:,:,k+1] + PV.T.values[:,:,k])*dpi
                    T_turbfluxdiv = -(PV.T.TurbFlux[:,:,k+1] - PV.T.TurbFlux[:,:,k])*dpi
                elif k==nl-1:
                    dwTdp_up =  0.5*DV.Wp.values[:,:,k]  *(PV.T.values[:,:,k] + PV.T.values[:,:,k-1])*dpi
                    dwTdp_dn = -0.5*DV.Wp.values[:,:,k+1]*(PV.T.values[:,:,k] + PV.T.values[:,:,k])*dpi
                    T_turbfluxdiv = -(T_sur_flux - PV.T.TurbFlux[:,:,k])*dpi
                else:
                    dwTdp_up =  0.5*DV.Wp.values[:,:,k]  *(PV.T.values[:,:,k]   + PV.T.values[:,:,k-1])*dpi
                    dwTdp_dn = -0.5*DV.Wp.values[:,:,k+1]*(PV.T.values[:,:,k+1] + PV.T.values[:,:,k])*dpi
                    T_turbfluxdiv = -(PV.T.TurbFlux[:,:,k+1] - PV.T.TurbFlux[:,:,k])*dpi
            else:
                dwTdp_dn = -0.5*wp[:,:,k+1]*(PV.T.values[:,:,k] + PV.T.values[:,:,k])*dpi
                T_turbfluxdiv = -(T_sur_flux - PV.T.TurbFlux[:,:,k])*dpi
            RHS_grid_T = np.add(np.add(np.add(np.add(T_turbfluxdiv, dwTdp_dn), dwTdp_up),PV.T.mp_tendency[:,:,k]),PV.T.forcing[:,:,k])
            RHS_grid_T = np.subtract(RHS_grid_T ,np.multiply(np.multiply(0.5/Pr.cp,np.multiply(np.add(DV.Wp.values[:,:,k+1],DV.Wp.values[:,:,k]),np.subtract(DV.gZ.values[:,:,k+1],DV.gZ.values[:,:,k]))),dpi))

            if Pr.moist_index > 0.0:
                uQT = np.multiply(DV.U.values[:,:,k], PV.QT.values[:,:,k])
                vQT = np.multiply(DV.V.values[:,:,k], PV.QT.values[:,:,k])
                if nl-1>0:
                    if k==0:
                        dwQTdp_dn = -0.5*DV.Wp.values[:,:,k+1]*(PV.QT.values[:,:,k+1] + PV.QT.values[:,:,k])*dpi
                        QT_turbfluxdiv = -(PV.QT.TurbFlux[:,:,k+1] - PV.QT.TurbFlux[:,:,k])*dpi
                    elif k==nl-1:
                        dwQTdp_up =  0.5*DV.Wp.values[:,:,k]  *(PV.QT.values[:,:,k] + PV.QT.values[:,:,k-1])*dpi
                        dwQTdp_dn = -0.5*DV.Wp.values[:,:,k+1]*(PV.QT.values[:,:,k] + PV.QT.values[:,:,k])*dpi
                        QT_turbfluxdiv = -(QT_sur_flux - PV.QT.TurbFlux[:,:,k])*dpi
                    else:
                        dwQTdp_up =  0.5*DV.Wp.values[:,:,k]  *(PV.QT.values[:,:,k]   + PV.QT.values[:,:,k-1])*dpi
                        dwQTdp_dn = -0.5*DV.Wp.values[:,:,k+1]*(PV.QT.values[:,:,k+1] + PV.QT.values[:,:,k])*dpi
                        QT_turbfluxdiv = -(PV.QT.TurbFlux[:,:,k+1] - PV.QT.TurbFlux[:,:,k])*dpi
                else:
                    dwQTdp_dn = -0.5*wp[:,:,k+1]*(PV.QT.values[:,:,k] + PV.QT.values[:,:,k])*dpi
                    QT_turbfluxdiv = -(QT_sur_flux - PV.QT.TurbFlux[:,:,k])*dpi
                RHS_grid_QT = np.add(np.add(np.add(np.add(QT_turbfluxdiv, dwQTdp_dn), dwQTdp_up),PV.QT.mp_tendency[:,:,k]),PV.QT.forcing[:,:,k])

            Dry_Energy_laplacian = Gr.laplacian*Gr.SphericalGrid.grdtospec(Dry_Energy)
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vorticity, v_vorticity)
            Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(uT, vT) # Vortical_T_flux is not used
            Vort_forc ,Div_forc = Gr.SphericalGrid.getvrtdivspec(DV.U.forcing[:,:,k],DV.V.forcing[:,:,k])
            w_vort_up ,w_div_up = Gr.SphericalGrid.getvrtdivspec(wu_up, wv_up)
            w_vort_dn ,w_div_dn = Gr.SphericalGrid.getvrtdivspec(wu_dn, wv_dn)
            RHS_T  = Gr.SphericalGrid.grdtospec(RHS_grid_T)

            if Pr.moist_index > 0.0:
                Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(uQT, vQT) # Vortical_T_flux is not used
                RHS_QT = Gr.SphericalGrid.grdtospec(RHS_grid_QT)

            PV.Vorticity.tendency[:,k]  = np.add(np.subtract(Vort_forc, np.add(np.add(Divergent_momentum_flux, w_vort_up), w_vort_dn)),
                                                 np.add(np.add(Vort_sur_flux, PV.Vorticity.ConvectiveFlux[:,k]), PV.Vorticity.sp_forcing[:,k]))
            PV.Divergence.tendency[:,k] = np.add(np.subtract(Vortical_momentum_flux, np.add(np.add(Dry_Energy_laplacian, w_div_up), w_div_dn)),
                                                np.add(np.add(Div_forc, Div_sur_flux), PV.Divergence.ConvectiveFlux[:,k]))
            PV.T.tendency[:,k] = np.add(np.subtract(RHS_T, Divergent_T_flux), PV.T.ConvectiveFlux[:,k])
            if Pr.moist_index > 0.0:
                PV.QT.tendency[:,k] = np.add(np.subtract(RHS_QT, Divergent_QT_flux), PV.QT.ConvectiveFlux[:,k])
        return
