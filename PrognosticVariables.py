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
        self.QT          = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Specific_humidity' ,  'qt','kg/kg' )
        self.P           = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,  'p','pasc' )
        return

    def initialize(self, Gr, DV):
        # need to define self.base_pressure as vector of n_layers
        self.Base_pressure = 100000.0
        self.T_init  = [229.0, 257.0, 295.0]
        self.P_init  = [Gr.p1, Gr.p2, Gr.p3, Gr.p_ref]
        self.QT_init = [0.0, 0.0, 0.0]

        self.Vorticity.values  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        self.Divergence.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        self.P.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers+1),  dtype=np.double, order='c'),self.P_init)
        self.T.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c'),self.T_init)
        self.QT.values         = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c'),self.QT_init)
        # initilize spectral values
        for k in range(Gr.n_layers):
            self.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(self.P.values[:,:,k])
            self.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(self.T.values[:,:,k])
            self.QT.spectral[:,k]          = Gr.SphericalGrid.grdtospec(self.QT.values[:,:,k])
            self.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(self.Vorticity.values[:,:,k])
            self.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(self.Divergence.values[:,:,k])
        self.P.spectral[:,Gr.n_layers]     = Gr.SphericalGrid.grdtospec(self.P.values[:,:,Gr.n_layers])

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
    # I needto define this function to ast on a general variable
    def physical_to_spectral(self, Gr):
        for k in range(Gr.n_layers):
            self.Vorticity.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Vorticity.values[:,:,k])
            self.Divergence.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values[:,:,k])
            self.T.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.T.values[:,:,k])
            self.QT.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.QT.values[:,:,k])
            # self.P.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,k])
        self.P.spectral[:,Gr.n_layers] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,Gr.n_layers])
        return

    # convert spectral data to spherical
    # I needto define this function to ast on a general variable
    def spectral_to_physical(self, Gr):
        for k in range(Gr.n_layers):
            self.Vorticity.values[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral[:,k])
            self.Divergence.values[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral[:,k])
            self.P.values[:,:,k]          = Gr.SphericalGrid.spectogrd(self.P.spectral[:,k])
            self.T.values[:,:,k]          = Gr.SphericalGrid.spectogrd(self.T.spectral[:,k])
            self.QT.values[:,:,k]         = Gr.SphericalGrid.spectogrd(self.QT.spectral[:,k])
        # I am updating only the surface pressure 
        self.P.values[:,:,Gr.n_layers] = Gr.SphericalGrid.spectogrd(self.P.spectral[:,Gr.n_layers])
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

    def reset_pressures(self, Gr):
        n=Gr.n_layers
        self.P.values[:,:,0] = np.add(np.zeros_like(self.P.values[:,:,0]),self.P_init[0])
        self.P.values[:,:,1] = np.add(np.zeros_like(self.P.values[:,:,1]),self.P_init[1])
        self.P.values[:,:,2] = np.add(np.zeros_like(self.P.values[:,:,2]),self.P_init[2])
        self.P.spectral[:,0] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,0])
        self.P.spectral[:,1] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,1])
        self.P.spectral[:,2] = Gr.SphericalGrid.grdtospec(self.P.values[:,:,2])
        return

    # this should be done in time intervals and save each time new files,not part of stats 
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

    def io(self, Gr, TS, Stats):
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Temperature',       self.T.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Gr, int(TS.t),1,           'Pressure',          self.P.values[:,:,Gr.n_layers])
        return

    def compute_tendencies(self, Gr, PV, DV, namelist):
        # # update DV
        # DV.Wp.values[:,:,0] = np.zeros_like(DV.Wp.values[:,:,0])
        # DV.gZ.values[:,:,3] = np.zeros_like(DV.Wp.values[:,:,0])
        # k=0
        # j = 2
        # DV.U.values[:,:,k], DV.V.values[:,:,k] = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,k],PV.Divergence.spectral[:,k])
        # DV.KE.values[:,:,k]    = 0.5*np.add(np.power(DV.U.values[:,:,k],2.0),np.power(DV.V.values[:,:,k],2.0))
        # DV.Wp.values[:,:,k+1]  = DV.Wp.values[:,:,k] + np.multiply(PV.P.values[:,:,k+1]-PV.P.values[:,:,k],PV.Divergence.values[:,:,k])
        # DV.gZ.values[:,:,j]    = np.multiply(Gr.Rd*PV.T.values[:,:,j],np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))) + DV.gZ.values[:,:,j+1]
        # k=1
        # j = 1
        # DV.U.values[:,:,k], DV.V.values[:,:,k] = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,k],PV.Divergence.spectral[:,k])
        # DV.KE.values[:,:,k]    = 0.5*np.add(np.power(DV.U.values[:,:,k],2.0),np.power(DV.V.values[:,:,k],2.0))
        # DV.Wp.values[:,:,k+1]  = DV.Wp.values[:,:,k]-np.multiply(PV.P.values[:,:,k+1]-PV.P.values[:,:,k],PV.Divergence.values[:,:,k])
        # DV.gZ.values[:,:,j]    = np.multiply(Gr.Rd*PV.T.values[:,:,j],np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))) + DV.gZ.values[:,:,j+1]
        # k=2
        # j = 0
        # DV.U.values[:,:,k], DV.V.values[:,:,k] = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,k],PV.Divergence.spectral[:,k])
        # DV.KE.values[:,:,k]    = 0.5*np.add(np.power(DV.U.values[:,:,k],2.0),np.power(DV.V.values[:,:,k],2.0))
        # DV.Wp.values[:,:,k+1]  = DV.Wp.values[:,:,k]-np.multiply(PV.P.values[:,:,k+1]-PV.P.values[:,:,k],PV.Divergence.values[:,:,k])
        # DV.gZ.values[:,:,j]    = np.multiply(Gr.Rd*PV.T.values[:,:,j],np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))) + DV.gZ.values[:,:,j+1]
        # DV.Wp.values[:,:,0]    = np.zeros_like(DV.Wp.values[:,:,0])

        # # update Fo
        # sigma_b = 0.7
        # k_a = (1.0/40.0)/(24.0*3600.0)
        # k_s = 0.25/(24.0*3600.0)
        # k_f = 1.0/(24.0*3600.0)

        # k=0
        # sigma_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
        # sigma_ratio_k = np.clip(np.divide(sigma_k-sigma_b,(1.0-sigma_b)) ,0.0, None)
        # cos4_lat = np.power(np.cos(Gr.lat),4.0)
        # k_T = np.multiply(np.multiply((k_s-k_a),sigma_ratio_k),cos4_lat)
        # k_T = np.add(k_a,k_T)
        # k_v = np.multiply(k_f,sigma_ratio_k)

        # Tbar_ = (315.0-60.0*np.sin(np.radians(Gr.lat))**2-10.0*
        #     np.log(PV.P.values[:,:,k]/Gr.p_ref)*np.cos(np.radians(Gr.lat))**2)*(PV.P.values[:,:,k]/Gr.p_ref)**Gr.kappa
        # Tbar = np.clip(Tbar_,200.0,350.0)

        # u_forcing = np.multiply(k_v,DV.U.values[:,:,k])
        # v_forcing = np.multiply(k_v,DV.V.values[:,:,k])
        # Vorticity_forcing, Divergece_forcing = Gr.SphericalGrid.getvrtdivspec(u_forcing, v_forcing)
        # PV.Divergence.forcing[:,k] = - Divergece_forcing
        # PV.Vorticity.forcing[:,k]  = - Vorticity_forcing
        # PV.T.forcing[:,k]          = - Gr.SphericalGrid.grdtospec(np.multiply(k_T,(PV.T.values[:,:,k] - Tbar)))

        # k=1
        # sigma_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
        # sigma_ratio_k = np.clip(np.divide(sigma_k-sigma_b,(1.0-sigma_b)) ,0.0, None)
        # cos4_lat = np.power(np.cos(Gr.lat),4.0)
        # k_T = np.multiply(np.multiply((k_s-k_a),sigma_ratio_k),cos4_lat)
        # k_T = np.add(k_a,k_T)
        # k_v = np.multiply(k_f,sigma_ratio_k)

        # Tbar_ = (315.0-60.0*np.sin(np.radians(Gr.lat))**2-10.0*
        #     np.log(PV.P.values[:,:,k]/Gr.p_ref)*np.cos(np.radians(Gr.lat))**2)*(PV.P.values[:,:,k]/Gr.p_ref)**Gr.kappa
        # Tbar = np.clip(Tbar_,200.0,350.0)

        # u_forcing = np.multiply(k_v,DV.U.values[:,:,k])
        # v_forcing = np.multiply(k_v,DV.V.values[:,:,k])
        # Vorticity_forcing, Divergece_forcing = Gr.SphericalGrid.getvrtdivspec(u_forcing, v_forcing)
        # PV.Divergence.forcing[:,k] = - np.multiply(Divergece_forcing, 0.0)
        # PV.Vorticity.forcing[:,k]  = - np.multiply(Vorticity_forcing, 0.0)
        # PV.T.forcing[:,k]          = - np.multiply(Gr.SphericalGrid.grdtospec(np.multiply(k_T,(PV.T.values[:,:,k] - Tbar))), 0.0)

        # k=2
        # sigma_k = np.divide(PV.P.values[:,:,k],PV.P.values[:,:,Gr.n_layers]) # as in josef's code for now
        # sigma_ratio_k = np.clip(np.divide(sigma_k-sigma_b,(1.0-sigma_b)) ,0.0, None)
        # cos4_lat = np.power(np.cos(Gr.lat),4.0)
        # k_T = np.multiply(np.multiply((k_s-k_a),sigma_ratio_k),cos4_lat)
        # k_T = np.add(k_a,k_T)
        # k_v = np.multiply(k_f,sigma_ratio_k)

        # Tbar_ = (315.0-60.0*np.sin(np.radians(Gr.lat))**2-10.0*
        #     np.log(PV.P.values[:,:,k]/Gr.p_ref)*np.cos(np.radians(Gr.lat))**2)*(PV.P.values[:,:,k]/Gr.p_ref)**Gr.kappa
        # Tbar = np.clip(Tbar_,200.0,350.0)

        # u_forcing = np.multiply(k_v,DV.U.values[:,:,k])
        # v_forcing = np.multiply(k_v,DV.V.values[:,:,k])
        # Vorticity_forcing, Divergece_forcing = Gr.SphericalGrid.getvrtdivspec(u_forcing, v_forcing)
        # PV.Divergence.forcing[:,k] = - np.multiply(Divergece_forcing, 0.0)
        # PV.Vorticity.forcing[:,k]  = - np.multiply(Vorticity_forcing, 0.0)
        # PV.T.forcing[:,k]          = - np.multiply(Gr.SphericalGrid.grdtospec(np.multiply(k_T,(PV.T.values[:,:,k] - Tbar))), 0.0)

        # bulku21=0.5*DV.Wp.values[:,:,1]*(DV.U.values[:,:,1]-DV.U.values[:,:,0])/(PV.P.values[:,:,1]-PV.P.values[:,:,0])
        # bulkv21=0.5*DV.Wp.values[:,:,1]*(DV.V.values[:,:,1]-DV.V.values[:,:,0])/(PV.P.values[:,:,1]-PV.P.values[:,:,0])
        # bulktemp21=0.5*DV.Wp.values[:,:,1]*(PV.T.values[:,:,1]+PV.T.values[:,:,0])/(PV.P.values[:,:,1]-PV.P.values[:,:,0])

        # bulku32=0.5*DV.Wp.values[:,:,2]*(DV.U.values[:,:,2]-DV.U.values[:,:,1])/(PV.P.values[:,:,2]-PV.P.values[:,:,1])
        # bulkv32=0.5*DV.Wp.values[:,:,2]*(DV.V.values[:,:,2]-DV.V.values[:,:,1])/(PV.P.values[:,:,2]-PV.P.values[:,:,1])
        # bulktemp32=0.5*DV.Wp.values[:,:,2]*(PV.T.values[:,:,2]+PV.T.values[:,:,1])/(PV.P.values[:,:,2]-PV.P.values[:,:,1])

        # bulku3s=0.5*DV.Wp.values[:,:,3]*(DV.U.values[:,:,3]-DV.U.values[:,:,2])/(PV.P.values[:,:,3]-PV.P.values[:,:,2])
        # bulkv3s=0.5*DV.Wp.values[:,:,3]*(DV.V.values[:,:,3]-DV.V.values[:,:,2])/(PV.P.values[:,:,3]-PV.P.values[:,:,2])
        # bulktemp3s=0.5*DV.Wp.values[:,:,3]*(PV.T.values[:,:,2]+PV.T.values[:,:,2])/(PV.P.values[:,:,3]-PV.P.values[:,:,2])
        # update PV
        # nz = Gr.n_layers
        # Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,2],np.subtract(PV.P.values[:,:,2],PV.P.values[:,:,3])),
        #                                                     np.multiply(DV.V.values[:,:,2],np.subtract(PV.P.values[:,:,2],PV.P.values[:,:,3]))) # Vortical_P_flux is not used
        # tmp_ = Divergent_P_flux + Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,2])
        # diver_sum = (np.multiply(PV.Divergence.spectral[:,1],np.subtract(PV.P.spectral[:,2],PV.P.spectral[:,1]))
        #             +np.multiply(PV.Divergence.spectral[:,0],np.subtract(PV.P.spectral[:,1],PV.P.spectral[:,0])))
        # PV.P.tendency[:,3] = (Divergent_P_flux + diver_sum)

        # if np.max(np.abs(PV.P.tendency[:,3]-tmp_))>1e-8:
        #     print('difference ', np.max(np.abs(PV.P.tendency[:,3]-tmp_)))
        #     print('new tendency', np.max(np.abs(PV.P.tendency[:,3])))
        #     print('old tendency', np.max(np.abs(tmp_)))
        #     print('diver_sum', np.max(np.abs(diver_sum)))
        #     print('W3', np.max(Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,2])))
        #     print('Divergent_P_flux =', np.max(np.abs(Divergent_P_flux)))

        # compute vertical fluxes for vorticity, divergence, temperature and specific humity
        # k=0
        # PV.T.VerticalFlux_k[:,:,k]  = np.divide(0.5*np.multiply(np.add(PV.T.values[:,:,k],PV.T.values[:,:,k+1]),DV.Wp.values[:,:,k+1]),
        #         PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
        # u_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(np.subtract(DV.U.values[:,:,k+1],DV.U.values[:,:,k]),
        #     np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])))
        # v_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(np.subtract(DV.V.values[:,:,k+1],DV.V.values[:,:,k]),
        #     np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])))
        # Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
        # PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
        # PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer
        
        # dp_ratio_ = np.multiply(0.0,PV.P.values[:,:,k])
        # dp_ratio = Gr.SphericalGrid.grdtospec(dp_ratio_)

        # # compute he energy laplacian for the divergence equation
        # Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
        #                        DV.gZ.values[:,:,k] + DV.KE.values[:,:,k])

        # # compute vorticity diveregnce componenets of horizontal momentum fluxes and temperature fluxes  - new name
        # Vorticity_flux, Divergence_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis),
        #                                       np.multiply(DV.V.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis))
        # Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
        #                                       np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
        # Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.QT.values[:,:,k]),
        #                                       np.multiply(DV.V.values[:,:,k],PV.QT.values[:,:,k])) # Vortical_QT_flux is not used


        # # compute thermal expension term for T equation
        # Thermal_expension = Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
        #                        - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp
        # # compute tendencies
        # PV.Divergence.tendency[:,k] = (Vorticity_flux - Dry_Energy_laplacian - PV.Divergence.sp_VerticalFlux[:,k]
        #                             - np.multiply(PV.Divergence.sp_VerticalFlux[:,k-1],dp_ratio)
        #                             + PV.Divergence.forcing[:,k])
        # PV.Vorticity.tendency[:,k]  = (- Divergence_flux - PV.Vorticity.sp_VerticalFlux[:,k]
        #                         - np.multiply(PV.Vorticity.sp_VerticalFlux[:,k-1],dp_ratio) - PV.Vorticity.forcing[:,k])
        # PV.T.tendency[:,k] = (- Divergent_T_flux + Gr.SphericalGrid.grdtospec(-PV.T.VerticalFlux[:,:,k])
        #                       - Thermal_expension  + PV.T.forcing[:,k])

        # tmp13 = DV.U.values[:,:,k]*(PV.T.values[:,:,k])
        # tmp14 = DV.v.values[:,:,k]*(PV.T.values[:,:,k])
        # tmpd1, tmpe1 = Gr.SphericalGrid.getvrtdivspec(tmp13,tmp14)
        # dtempsp1 = -tmpe1 +Gr.SphericalGrid.grdtospec(-np.multiply(omega2,(phi2-phi1)/(p2-p1)) -bulktemp21 +fT1)
        
        # PV.QT.tendency[:,k]          = np.zeros_like(PV.QT.spectral[:,k])


        # k=1
        # PV.T.VerticalFlux[:,:,k]  = np.divide(0.5*np.multiply(np.add(PV.T.values[:,:,k],PV.T.values[:,:,k+1]),DV.Wp.values[:,:,k+1]),
        #         PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
        # u_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(np.subtract(DV.U.values[:,:,k+1], DV.U.values[:,:,k]),
        #     np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])))
        # v_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(np.subtract(DV.V.values[:,:,k+1], DV.V.values[:,:,k]),
        #     np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])))
        # Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
        # PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
        # PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer
        
        # dp_ratio_ = (PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
        # dp_ratio = Gr.SphericalGrid.grdtospec(dp_ratio_)

        # # compute he energy laplacian for the divergence equation
        # Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
        #                        DV.gZ.values[:,:,k] + DV.KE.values[:,:,k])

        # # compute vorticity diveregnce componenets of horizontal momentum fluxes and temperature fluxes  - new name
        # Vorticity_flux, Divergence_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis),
        #                                       np.multiply(DV.V.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis))
        # Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
        #                                       np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
        # Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.QT.values[:,:,k]),
        #                                       np.multiply(DV.V.values[:,:,k],PV.QT.values[:,:,k])) # Vortical_QT_flux is not used


        # # compute thermal expension term for T equation
        # Thermal_expension = Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
        #                        - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp
        # # compute tendencies
        # PV.Divergence.tendency[:,k] = (Vorticity_flux - Dry_Energy_laplacian - PV.Divergence.sp_VerticalFlux[:,k]
        #                             - np.multiply(PV.Divergence.sp_VerticalFlux[:,k-1],dp_ratio)
        #                             + PV.Divergence.forcing[:,k])
        # PV.Vorticity.tendency[:,k]  = (- Divergence_flux - PV.Vorticity.sp_VerticalFlux[:,k]
        #                         - np.multiply(PV.Vorticity.sp_VerticalFlux[:,k-1],dp_ratio) - PV.Vorticity.forcing[:,k])
        # PV.T.tendency[:,k] = (- Divergent_T_flux + Gr.SphericalGrid.grdtospec(-PV.T.VerticalFlux[:,:,k] + np.multiply(PV.T.VerticalFlux[:,:,k-1],dp_ratio_))
        #                       - Thermal_expension  + PV.T.forcing[:,k])
        # # PV.QT.tendency[:,k] = - Divergent_QT_flux + Gr.SphericalGrid.grdtospec(-PV.QT.VerticalFlux[:,:,k] + np.multiply(PV.QT.VerticalFlux[:,:,k-1],dp_ratio_)) + PV.QT.forcing[:,k]
        # PV.QT.tendency[:,k]          = np.zeros_like(PV.QT.spectral[:,k])


        # k=2
        # PV.T.VerticalFlux[:,:,k]  = np.multiply(0.0,DV.Wp.values[:,:,k+1])
        # PV.QT.VerticalFlux[:,:,k] = np.multiply(0.0,DV.Wp.values[:,:,k+1])
        # u_vertical_flux = np.multiply(0.0,DV.Wp.values[:,:,k+1])
        # v_vertical_flux = np.multiply(0.0,DV.Wp.values[:,:,k+1])
        # Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
        # PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
        # PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer

        # dp_ratio_ = (PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
        # dp_ratio = Gr.SphericalGrid.grdtospec(dp_ratio_)
        

        # # compute he energy laplacian for the divergence equation
        # Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
        #                        DV.gZ.values[:,:,k] + DV.KE.values[:,:,k])

        # # compute vorticity diveregnce componenets of horizontal momentum fluxes and temperature fluxes  - new name
        # Vorticity_flux, Divergence_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis),
        #                                       np.multiply(DV.V.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis))
        # Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
        #                                       np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
        # Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.QT.values[:,:,k]),
        #                                       np.multiply(DV.V.values[:,:,k],PV.QT.values[:,:,k])) # Vortical_QT_flux is not used


        # # compute thermal expension term for T equation
        # Thermal_expension = Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
        #                        - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp
        # # compute tendencies
        # PV.Divergence.tendency[:,k] = (Vorticity_flux - Dry_Energy_laplacian - PV.Divergence.sp_VerticalFlux[:,k]
        #                             - np.multiply(PV.Divergence.sp_VerticalFlux[:,k-1],dp_ratio)
        #                             + PV.Divergence.forcing[:,k])
        # PV.Vorticity.tendency[:,k]  = (- Divergence_flux - PV.Vorticity.sp_VerticalFlux[:,k]
        #                         - np.multiply(PV.Vorticity.sp_VerticalFlux[:,k-1],dp_ratio) - PV.Vorticity.forcing[:,k])
        # PV.T.tendency[:,k] = (- Divergent_T_flux + Gr.SphericalGrid.grdtospec(-PV.T.VerticalFlux[:,:,k] + np.multiply(PV.T.VerticalFlux[:,:,k-1],dp_ratio_))
        #                       - Thermal_expension  + PV.T.forcing[:,k])
        # # PV.QT.tendency[:,k] = - Divergent_QT_flux + Gr.SphericalGrid.grdtospec(-PV.QT.VerticalFlux[:,:,k] + np.multiply(PV.QT.VerticalFlux[:,:,k-1],dp_ratio_)) + PV.QT.forcing[:,k]
        # PV.QT.tendency[:,k]          = np.zeros_like(PV.QT.spectral[:,k])

        # yair stopped here
        # temp1 = Gr.SphericalGrid.spectogrd(PV.T.spectral[:,0])
        # temp2 = Gr.SphericalGrid.spectogrd(PV.T.spectral[:,1])
        # temp3 = Gr.SphericalGrid.spectogrd(PV.T.spectral[:,2])
        temp1 = PV.T.values[:,:,0]
        temp2 = PV.T.values[:,:,1]
        temp3 = PV.T.values[:,:,2]

        # u1, v1 = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,0],PV.Divergence.spectral[:,0])
        # u2, v2 = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,1],PV.Divergence.spectral[:,1])
        # u3, v3 = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,2],PV.Divergence.spectral[:,2])

        u1 = DV.U.values[:,:,0]
        v1 = DV.V.values[:,:,0]
        u2 = DV.U.values[:,:,1]
        v2 = DV.V.values[:,:,1]
        u3 = DV.U.values[:,:,2]
        v3 = DV.V.values[:,:,2]

        p1 = 25000.0#PV.P.values[:,:,0]
        p2 = 50000.0#PV.P.values[:,:,1]
        p3 = 85000.0#PV.P.values[:,:,2]
        ps = Gr.SphericalGrid.spectogrd(PV.P.spectral[:,3])

        p1_sp = 25000.0#PV.P.spectral[:,0]
        p2_sp = 50000.0#PV.P.spectral[:,1]
        p3_sp = 85000.0#PV.P.spectral[:,2]
        ps_sp = PV.P.spectral[:,3]

        vrt1 = PV.Vorticity.values[:,:,0]
        vrt2 = PV.Vorticity.values[:,:,1]
        vrt3 = PV.Vorticity.values[:,:,2]
        div1 = PV.Divergence.values[:,:,0]
        div2 = PV.Divergence.values[:,:,1]
        div3 = PV.Divergence.values[:,:,2]

        vrtsp1 = PV.Vorticity.spectral[:,0]
        vrtsp2 = PV.Vorticity.spectral[:,1]
        vrtsp3 = PV.Vorticity.spectral[:,2]
        divsp1 = PV.Divergence.spectral[:,0]
        divsp2 = PV.Divergence.spectral[:,1]
        divsp3 = PV.Divergence.spectral[:,2]

        y=Gr.lat[:,0]
        p0 = Gr.p_ref
        sigma_b=0.7      # sigma coordiantes as sigma=p/ps
        k_a = 1./40./(3600.*24)     # [1/s]
        k_b = 1./10./(3600.*24)     # [1/s]
        k_s = 1./4./(3600.*24)      # [1/s]
        k_f = 1./(3600.*24)         # [1/s]
        DT_y= 60.        # Characteristic temperature change in meridional direction [K]
        Dtheta_z = 10.   # Characteristic potential temperature change in vertical [K]
        Tbar1=np.zeros_like(temp1)
        Tbar2=np.zeros_like(temp2)
        Tbar3=np.zeros_like(temp3)
        p21=p2-p1
        p12=p2+p1
        p32=p3-p2
        p23=p3+p2

        # for jj in np.arange(0,Gr.nlons,1):
        #     Tbar1[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z*np.log((p1[:,jj]+p2[:,jj])/(2.*p0))*np.cos(y)**2)*((p1[:,jj]+p2[:,jj])/(2.*p0))**Gr.kappa
        #     Tbar2[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z*np.log((p2[:,jj]+p3[:,jj])/(2.*p0))*np.cos(y)**2)*((p2[:,jj]+p3[:,jj])/(2.*p0))**Gr.kappa
        #     Tbar3[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z*np.log((p3[:,jj]+ps[:,jj])/(2.*p0))*np.cos(y)**2)*((p3[:,jj]+ps[:,jj])/(2.*p0))**Gr.kappa
        for jj in np.arange(0,Gr.nlons,1):
            Tbar1[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z*np.log((p1+p2)/(2.*p0))*np.cos(y)**2)*((p1+p2)/(2.*p0))**Gr.kappa
            Tbar2[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z*np.log((p2+p3)/(2.*p0))*np.cos(y)**2)*((p2+p3)/(2.*p0))**Gr.kappa
            Tbar3[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z*np.log((p3+ps[:,jj])/(2.*p0))*np.cos(y)**2)*((p3+ps[:,jj])/(2.*p0))**Gr.kappa
        
        Tbar1[Tbar1<=200.]=200. # minimum equilibrium Temperature is 200 K
        Tbar2[Tbar2<=200.]=200. # minimum equilibrium Temperature is 200 K
        Tbar3[Tbar3<=200.]=200. # minimum equilibrium Temperature is 200 K
        # Relaxation time scales
        # May be increase time scales to 1000 or 2000 days to see the formation of jets
        # But make sure to run the simulation till at least twice this value
        #run 1000 days
        # from Held&Suarez (1994)
        sigma1=np.divide((p1+p2)/2.,ps)
        sigma2=np.divide((p2+p3)/2.,ps)
        sigma3=np.divide((p3+ps)/2.,ps)
        #sigma1=np.divide(p1,ps)
        #sigma2=np.divide(p2,ps)
        #sigma3=np.divide(p3,ps)
        #sigma1=np.divide(p1,p0)
        #sigma2=np.divide(p2,p0)
        #sigma3=np.divide(p3,p0)
        sigma_ratio1=np.clip(np.divide(sigma1-sigma_b,1-sigma_b),0,None)
        sigma_ratio2=np.clip(np.divide(sigma2-sigma_b,1-sigma_b),0,None)
        sigma_ratio3=np.clip(np.divide(sigma3-sigma_b,1-sigma_b),0,None)
        k_T1=k_a+(k_s-k_a)*np.multiply(sigma_ratio1,np.power(np.cos(Gr.lat),4))
        k_T2=k_a+(k_s-k_a)*np.multiply(sigma_ratio2,np.power(np.cos(Gr.lat),4))
        k_T3=k_a+(k_s-k_a)*np.multiply(sigma_ratio3,np.power(np.cos(Gr.lat),4))
        k_v1=k_b+k_f*sigma_ratio1
        k_v2=k_b+k_f*sigma_ratio2
        k_v3=k_b+k_f*sigma_ratio3
        #

        #exit()

        # ### level 3 ###
        # vrt3 = Gr.SphericalGrid.spectogrd(vrtsp3)
        # div3 = Gr.SphericalGrid.spectogrd(divsp3)
        # u3,v3 = Gr.SphericalGrid.getuv(vrtsp3,divsp3)
        # temp3  = Gr.SphericalGrid.spectogrd(tempsp3)
        # ### level 2 ###
        # vrt2 = Gr.SphericalGrid.spectogrd(vrtsp2)
        # div2 = Gr.SphericalGrid.spectogrd(divsp2)
        # u2,v2 = Gr.SphericalGrid.getuv(vrtsp2,divsp2)
        # temp2  = Gr.SphericalGrid.spectogrd(tempsp2)
        # ### level 1 ###
        # vrt1 = Gr.SphericalGrid.spectogrd(vrtsp1)
        # div1 = Gr.SphericalGrid.spectogrd(divsp1)
        # u1,v1 = Gr.SphericalGrid.getuv(vrtsp1,divsp1)
        # temp1  = Gr.SphericalGrid.spectogrd(tempsp1)


        #
        # quantities across levels
        # omega2=(p1-p2)*div1
        # omega3=omega2+(p2-p3)*div2
        # omegas=omega3+(p3-ps)*div3
        # omega2=(PV.P.values[:,:,0]-PV.P.values[:,:,1])*PV.Divergence.values[:,:,0]
        # omega3=omega2+(PV.P.values[:,:,1]-PV.P.values[:,:,2])*PV.Divergence.values[:,:,1]
        # omegas=omega3+(PV.P.values[:,:,2]-PV.P.values[:,:,3])*PV.Divergence.values[:,:,2]
        omega2=DV.Wp.values[:,:,1]
        omega3=DV.Wp.values[:,:,2]
        omegas=DV.Wp.values[:,:,3]
        #
        omegasp2=Gr.SphericalGrid.grdtospec(omega2)
        omegasp3=Gr.SphericalGrid.grdtospec(omega3)
        omegasps=Gr.SphericalGrid.grdtospec(omegas)

        #ps_div = div((u3,v3)*(p3-ps))
        tmp1=u3*(p3-ps)
        tmp2=v3*(p3-ps)
        ps_vrt, ps_div = Gr.SphericalGrid.getvrtdivspec(tmp1, tmp2)

        #surface pressure
        dpssp=ps_div + omegasp3

        #exchange terms (vertical)
        # bulku21=0.5*omega2*(u2-u1)/p21
        # bulkv21=0.5*omega2*(v2-v1)/p21
        # bulktemp21=0.5*omega2*(temp2+temp1)/p21
        bulku21=0.5*np.multiply(omega2,(u2-u1)/p21)
        bulkv21=0.5*np.multiply(omega2,(v2-v1)/p21)
        bulktemp21=0.5*np.multiply(omega2,(temp2+temp1)/p21)


        # bulku32=0.5*omega3*(u3-u2)/p32
        # bulkv32=0.5*omega3*(v3-v2)/p32
        # bulktemp32=0.5*omega3*(temp3+temp2)/p32
        bulku32=0.5*np.multiply(omega3,(u3-u2)/p32)
        bulkv32=0.5*np.multiply(omega3,(v3-v2)/p32)
        bulktemp32=0.5*np.multiply(omega3,(temp3+temp2)/p32)

        # bulku3s=0.5*omega3*(u3-u2)/(ps-p3)
        # bulkv3s=0.5*omega3*(v3-v2)/(ps-p3)
        # bulktemp3s=omegas*temp3/(ps-p3)
        bulku3s=0.5*np.multiply(omegas,(u3-u2)/(ps-p3))
        bulkv3s=0.5*np.multiply(omegas,(v3-v2)/(ps-p3))
        bulktemp3s=0.5*np.multiply(omegas,(temp3+temp2)/(ps-p3))


        vrtsp21, divsp21 = Gr.SphericalGrid.getvrtdivspec(bulku21, bulkv21)
        vrtsp32, divsp32 = Gr.SphericalGrid.getvrtdivspec(bulku32, bulkv32)
        vrtsp3s, divsp3s = Gr.SphericalGrid.getvrtdivspec(bulku3s, bulkv3s)

        #wind forcing from H&S
        fu1=-np.multiply(k_v1,u1)
        fu2=-np.multiply(k_v2,u2)
        fu3=-np.multiply(k_v3,u3)
        fv1=-np.multiply(k_v1,v1)
        fv2=-np.multiply(k_v2,v2)
        fv3=-np.multiply(k_v3,v3)
        fvrtsp1, fdivsp1 = Gr.SphericalGrid.getvrtdivspec(fu1, fv1)
        fvrtsp2, fdivsp2 = Gr.SphericalGrid.getvrtdivspec(fu2, fv2)
        fvrtsp3, fdivsp3 = Gr.SphericalGrid.getvrtdivspec(fu3, fv3)

        #temperature forcing
        fT1=-np.multiply(k_T1,(temp1-Tbar1))
        fT2=-np.multiply(k_T2,(temp2-Tbar2))
        fT3=-np.multiply(k_T3,(temp3-Tbar3))


        # geopotentials
        # phi3 =Gr.Rd*temp3*np.log(ps/p3)
        # phi2 =Gr.Rd*temp2*np.log(p3/p2)  + phi3
        # phi1 =Gr.Rd*temp1*np.log(p2/p1)  + phi2

        phi3 =Gr.Rd*PV.T.values[:,:,2]*np.log(ps/PV.P.values[:,:,2])
        phi2 =Gr.Rd*PV.T.values[:,:,1]*np.log(p3/PV.P.values[:,:,1])  + phi3
        phi1 =Gr.Rd*PV.T.values[:,:,0]*np.log(p2/PV.P.values[:,:,0])  + phi2


        # kinetic energy
        # ekin3=0.5*(u3**2+v3**2)
        # ekin1=0.5*(u1**2+v1**2)
        # ekin2=0.5*(u2**2+v2**2)
        ekin1=DV.KE.values[:,:,0]
        ekin2=DV.KE.values[:,:,1]
        ekin3=DV.KE.values[:,:,2]





        ### level 3 ###
        tmp31 = u3*(vrt3+Gr.Coriolis)
        tmp32 = v3*(vrt3+Gr.Coriolis)
        tmpa3, tmpb3 = Gr.SphericalGrid.getvrtdivspec(tmp31, tmp32)
        # F3 = fr.forcingFn(F0)
        dvrtsp3 = -tmpb3  -vrtsp3s +fvrtsp3
        #dvrtsp3 += Gr.SphericalGrid.lap*F3*.1
        #
        tmp33 = u3*(temp3)
        tmp34 = v3*(temp3)
        tmpd3, tmpe3 = Gr.SphericalGrid.getvrtdivspec(tmp33,tmp34)
        dtempsp3 = -tmpe3 +Gr.SphericalGrid.grdtospec(np.multiply(omegas,np.divide(phi3,(ps-p3)))/Gr.cp -bulktemp3s + np.divide(bulktemp32*(p3-p2),(ps-p3)) +fT3) 
        #
        tmpf3 = Gr.SphericalGrid.grdtospec(phi3+ekin3)
        #tmpf3 = Gr.SphericalGrid.grdtospec(phi3m+ekin3)
        ddivsp3 = tmpa3 - Gr.SphericalGrid.lap*tmpf3 -divsp3s +fdivsp3
        #




        ### level 2 ###
        tmp21 = u2*(vrt2+Gr.Coriolis)
        tmp22 = v2*(vrt2+Gr.Coriolis)
        tmpa2, tmpb2 = Gr.SphericalGrid.getvrtdivspec(tmp21, tmp22)
        # F2 = fr.forcingFn(F0)
        p1_ = 25000.0
        p2_ = 50000.0
        p3_ = 85000.0
        dvrtsp2 = -tmpb2 -vrtsp32  -vrtsp21*(p2_-p1_)/(p3_-p2_) +fvrtsp2 
        #dvrtsp2 += Gr.SphericalGrid.lap*F2*.15
        #
        tmp23 = u2*(temp2)
        tmp24 = v2*(temp2)
        tmpd2, tmpe2 = Gr.SphericalGrid.getvrtdivspec(tmp23,tmp24)
        dtempsp2 = -tmpe2 +Gr.SphericalGrid.grdtospec(-np.multiply(omega3,(phi3-phi2)/(p3-p2))/Gr.cp -bulktemp32 +bulktemp21*(p2-p1)/(p3-p2) + fT2)  
        #
        tmpf2 = Gr.SphericalGrid.grdtospec(phi2+ekin2)
        #tmpf2 = Gr.SphericalGrid.grdtospec(phi2m+ekin2)
        ddivsp2 = tmpa2 - Gr.SphericalGrid.lap*tmpf2  -divsp32 -divsp21*(p2_-p1_)/(p3_-p2_)  +fdivsp2 





        ### level 1 ###
        tmp11 = u1*(vrt1+Gr.Coriolis)
        tmp12 = v1*(vrt1+Gr.Coriolis)
        tmpa1, tmpb1 = Gr.SphericalGrid.getvrtdivspec(tmp11, tmp12)
        # F1 = fr.forcingFn(F0)
        dvrtsp1 = -tmpb1 -vrtsp21 +fvrtsp1
        #dvrtsp1 = -tmpb1 -vrtsp21  
        #dvrtsp1 += Gr.SphericalGrid.lap*F1*.2
        #
        tmp13 = u1*(temp1)
        tmp14 = v1*(temp1)
        tmpd1, tmpe1 = Gr.SphericalGrid.getvrtdivspec(tmp13,tmp14)
        dtempsp1 = -tmpe1 +Gr.SphericalGrid.grdtospec(-np.multiply(omega2,(phi2-phi1)/(p2-p1))/Gr.cp -bulktemp21 +fT1)  
        #
        tmpf1 = Gr.SphericalGrid.grdtospec(phi1+ekin1)
        #tmpf1 = Gr.SphericalGrid.grdtospec(phi1m+ekin1)
        ddivsp1 = tmpa1 -Gr.SphericalGrid.lap*tmpf1  -divsp21 +fdivsp1 
        #ddivsp1 = tmpa1 -Gr.SphericalGrid.lap*tmpf1 -divsp21  
        #
        k=0
        PV.Divergence.tendency[:,k] = ddivsp1
        PV.Vorticity.tendency[:,k]  = dvrtsp1
        PV.T.tendency[:,k] = dtempsp1
        k=1
        PV.Divergence.tendency[:,k] = ddivsp2
        PV.Vorticity.tendency[:,k]  = dvrtsp2
        PV.T.tendency[:,k] = dtempsp2
        k=2
        PV.Divergence.tendency[:,k] = ddivsp3
        PV.Vorticity.tendency[:,k]  = dvrtsp3
        PV.T.tendency[:,k] = dtempsp3

        PV.P.tendency[:,3] = dpssp


        # for k in range(nz):
        #     T_high  = PV.T.values[:,:,k]
        #     QT_high = PV.QT.values[:,:,k]
        #     U_high  = DV.U.values[:,:,k]
        #     V_high  = DV.V.values[:,:,k]
        #     if k==nz-1:
        #         T_low  = T_high
        #         QT_low = QT_high
        #         U_low  = U_high
        #         V_low  = V_high
        #     else:
        #         T_low  = PV.T.values[:,:,k+1]
        #         QT_low = PV.QT.values[:,:,k+1]
        #         U_low  = DV.U.values[:,:,k+1]
        #         V_low  = DV.V.values[:,:,k+1]
        #     # as a convention the vertical flux at k is comming ot the layer k from below (k+1)
        #     PV.T.VerticalFlux[:,:,k]  = np.divide(0.5*np.multiply(T_high+T_low,DV.Wp.values[:,:,k+1]),
        #             PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
        #     PV.QT.VerticalFlux[:,:,k] = np.divide(0.5*np.multiply(QT_high+QT_low,DV.Wp.values[:,:,k+1]),
        #             PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
        #     u_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(U_low - U_high,
        #         PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))
        #     v_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(V_low - V_high,
        #         PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))
        #     Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
        #     PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
        #     PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer

        # for k in range(nz):
        #     # dp_ratio is the ratio of dp between the layers used for correction of the vertical fluxes
        #     if k==0:
        #         dp_ratio_ = np.multiply(0.0,PV.P.values[:,:,k])
        #     else:
        #         dp_ratio_ = (PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
        #     dp_ratio = Gr.SphericalGrid.grdtospec(dp_ratio_)
        #     # compute he energy laplacian for the divergence equation
        #     Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
        #                            DV.gZ.values[:,:,k] + DV.KE.values[:,:,k])

        #     # compute vorticity diveregnce componenets of horizontal momentum fluxes and temperature fluxes  - new name
        #     Vorticity_flux, Divergence_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis),
        #                                           np.multiply(DV.V.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis))
        #     Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
        #                                           np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
        #     Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.QT.values[:,:,k]),
        #                                           np.multiply(DV.V.values[:,:,k],PV.QT.values[:,:,k])) # Vortical_QT_flux is not used


        #     # compute thermal expension term for T equation
        #     Thermal_expension = Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
        #                            - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp
        #     # compute tendencies
        #     PV.Divergence.tendency[:,k] = (Vorticity_flux - Dry_Energy_laplacian - PV.Divergence.sp_VerticalFlux[:,k]
        #                                 - np.multiply(PV.Divergence.sp_VerticalFlux[:,k-1],dp_ratio)
        #                                 + PV.Divergence.forcing[:,k])
        #     PV.Vorticity.tendency[:,k]  = (- Divergence_flux - PV.Vorticity.sp_VerticalFlux[:,k]
        #                             - np.multiply(PV.Vorticity.sp_VerticalFlux[:,k-1],dp_ratio) - PV.Vorticity.forcing[:,k])
        #     PV.T.tendency[:,k] = (- Divergent_T_flux + Gr.SphericalGrid.grdtospec(-PV.T.VerticalFlux[:,:,k] + np.multiply(PV.T.VerticalFlux[:,:,k-1],dp_ratio_))
        #                           - Thermal_expension  + PV.T.forcing[:,k])
        #     # PV.QT.tendency[:,k] = - Divergent_QT_flux + Gr.SphericalGrid.grdtospec(-PV.QT.VerticalFlux[:,:,k] + np.multiply(PV.QT.VerticalFlux[:,:,k-1],dp_ratio_)) + PV.QT.forcing[:,k]
        #     PV.QT.tendency[:,k]          = np.zeros_like(PV.QT.spectral[:,k])

        return