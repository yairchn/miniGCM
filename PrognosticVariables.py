
import matplotlib.pyplot as plt
import scipy as sc
import netCDF4
import seaborn
import numpy as np
from math import *

class PrognosticVariable:
    def __init__(self, nx, ny, nl, n_spec, kind, name, units):
        self.values = np.zeros((nx,ny,nl),dtype=np.double, order='c')
        self.SurfaceFlux = np.zeros((nx,ny),dtype=np.double, order='c')
        self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.double, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.old = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.now = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.tendency = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.sp_VerticalFlux = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.forcing = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.kind = kind
        self.name = name
        self.units = units
        return

class PrognosticVariables:
    def __init__(self, Gr):
        self.Vorticity   = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'vorticity' , 'zeta',  '1/s' )
        self.Divergence  = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'divergance', 'delta', '1/s' )
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
        self.Vorticity.old  = self.Vorticity.now
        self.Divergence.old = self.Divergence.now
        self.T.old          = self.T.now
        self.QT.old         = self.QT.now
        self.P.old          = self.P.now
        return

    def set_now_with_tendencies(self):
        self.Vorticity.now  = self.Vorticity.tendency
        self.Divergence.now = self.Divergence.tendency
        self.T.now          = self.T.tendency
        self.QT.now         = self.QT.tendency
        self.P.now          = self.P.tendency
        return

    # this should be done in time intervals and save each time new files,not part of stats 
    def stats_io(self, TS, Stats):
        Stats.write_global_mean('global_mean_T', self.T.values, TS.t)
        Stats.write_global_mean('global_mean_QT', self.QT.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_P',self.P.values[:,:,1:4], TS.t)
        Stats.write_zonal_mean('zonal_mean_T',self.T.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_QT',self.QT.values, TS.t)
        return

    def io(self, Gr, TS, Stats):
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Temperature',       self.T.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Gr, int(TS.t),1,           'Pressure',          self.P.values[:,:,Gr.n_layers])
        return

    def compute_tendencies(self, Gr, PV, DV, namelist):
        nz = Gr.n_layers
        # compute surface pressure tendency
        Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,nz-1],PV.P.values[:,:,nz-1]-PV.P.values[:,:,nz]),
                                                            np.multiply(DV.V.values[:,:,nz-1],PV.P.values[:,:,nz-1]-PV.P.values[:,:,nz])) # Vortical_P_flux is not used
        PV.P.tendency[:,nz] = Divergent_P_flux + DV.Wp.spectral[:,nz-1]

        # print(np.min(PV.P.tendency[:,nz+1]))
        # print(np.max(PV.P.tendency[:,nz+1]))
        # plt.figure('omega 1')
        # plt.contourf(DV.Wp.values[:,:,0])
        # plt.colorbar()
        # plt.figure('omega 2')
        # plt.contourf(DV.Wp.values[:,:,1])
        # plt.colorbar()
        # plt.figure('omega 3')
        # plt.contourf(DV.Wp.values[:,:,2])
        # plt.colorbar()
        # plt.figure('omega s')
        # plt.contourf(DV.Wp.values[:,:,3])
        # plt.colorbar()

        # plt.show()
        # plt.figure('ps tendency')
        # plt.contourf(Gr.SphericalGrid.spectogrd(PV.P.tendency[:,nz]))
        # plt.colorbar()
        # plt.figure('Divergent_P_flux')
        # plt.contourf(Gr.SphericalGrid.spectogrd(Divergent_P_flux))
        # plt.colorbar()
        # plt.figure('omega 3')
        # plt.contourf(Gr.SphericalGrid.spectogrd(DV.Wp.spectral[:,nz-1]))
        # plt.colorbar()
        # plt.show()

        # compute vertical fluxes for vorticity, divergence, temperature and specific humity
        for k in range(nz):
            T_high  = PV.T.values[:,:,k]
            QT_high = PV.QT.values[:,:,k]
            U_high  = DV.U.values[:,:,k]
            V_high  = DV.V.values[:,:,k]
            if k==nz-1:
                T_low  = T_high
                QT_low = QT_high
                U_low  = U_high
                V_low  = V_high
            else:
                T_low  = PV.T.values[:,:,k+1]
                QT_low = PV.QT.values[:,:,k+1]
                U_low  = DV.U.values[:,:,k+1]
                V_low  = DV.V.values[:,:,k+1]
            # as a convention the vertical flux at k is comming ot the layer k from below (k+1)
            PV.T.VerticalFlux[:,:,k]  = np.divide(0.5*np.multiply(T_high+T_low,DV.Wp.values[:,:,k+1]),
                    PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
            PV.QT.VerticalFlux[:,:,k] = np.divide(0.5*np.multiply(QT_high+QT_low,DV.Wp.values[:,:,k+1]),
                    PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
            u_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(U_low - U_high,
                PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))
            v_vertical_flux = np.multiply(0.5*DV.Wp.values[:,:,k+1],np.divide(V_low - V_high,
                PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
            PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
            PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer

        for k in range(nz):
            # dp_ratio is the ratio of dp between the layers used for correction of the vertical fluxes
            if k==0:
                dp_ratio_ = np.multiply(0.0,PV.P.values[:,:,k])
            else:
                dp_ratio_ = (PV.P.values[:,:,k]-PV.P.values[:,:,k-1])/(PV.P.values[:,:,k+1]-PV.P.values[:,:,k])
            dp_ratio = Gr.SphericalGrid.grdtospec(dp_ratio_)
            # compute he energy laplacian for the divergence equation
            Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
                                   DV.gZ.values[:,:,k] + DV.KE.values[:,:,k])

            # compute vorticity diveregnce componenets of horizontal momentum fluxes and temperature fluxes  - new name
            Vorticity_flux, Divergence_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis),
                                                  np.multiply(DV.V.values[:,:,k],PV.Vorticity.values[:,:,k]+Gr.Coriolis))
            Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
                                                  np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
            Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.QT.values[:,:,k]),
                                                  np.multiply(DV.V.values[:,:,k],PV.QT.values[:,:,k])) # Vortical_QT_flux is not used


            # compute thermal expension term for T equation
            Thermal_expension = Gr.SphericalGrid.grdtospec(DV.Wp.values[:,:,k+1]*(DV.gZ.values[:,:,k+1]
                                   - DV.gZ.values[:,:,k])/(PV.P.values[:,:,k+1] - PV.P.values[:,:,k]))/Gr.cp
            # compute tendencies
            PV.Divergence.tendency[:,k] = (Vorticity_flux - Dry_Energy_laplacian - PV.Divergence.sp_VerticalFlux[:,k]
                                        - np.multiply(PV.Divergence.sp_VerticalFlux[:,k-1],dp_ratio)
                                        + PV.Divergence.forcing[:,k])
            PV.Vorticity.tendency[:,k]  = (- Divergence_flux - PV.Vorticity.sp_VerticalFlux[:,k]
                                    - np.multiply(PV.Vorticity.sp_VerticalFlux[:,k-1],dp_ratio) - PV.Vorticity.forcing[:,k])
            PV.T.tendency[:,k] = (- Divergent_T_flux + Gr.SphericalGrid.grdtospec(-PV.T.VerticalFlux[:,:,k] + np.multiply(PV.T.VerticalFlux[:,:,k-1],dp_ratio_))
                                  - Thermal_expension  + PV.T.forcing[:,k])
            PV.QT.tendency[:,k] = - Divergent_QT_flux + Gr.SphericalGrid.grdtospec(-PV.QT.VerticalFlux[:,:,k] + np.multiply(PV.QT.VerticalFlux[:,:,k-1],dp_ratio_)) + PV.QT.forcing[:,k]
            # print("=====================================================================")
            # print('k= ',k)
            # print('Vorticity_flux', Vorticity_flux)
            # print('Dry_Energy_laplacian', Dry_Energy_laplacian)
            # print('PV.Divergence.sp_VerticalFlux[:,k]', PV.Divergence.sp_VerticalFlux[:,k])
            # print('np.multiply(PV.Divergence.sp_VerticalFlux[:,k],dp_ratio)', np.multiply(PV.Divergence.sp_VerticalFlux[:,k],dp_ratio))
            # print('PV.Divergence.forcing[:,k]', PV.Divergence.forcing[:,k])

            # print('Divergent_T_flux',np.max(Divergent_T_flux),np.min(Divergent_T_flux))
            # print('VerticalFlux',np.max(np.multiply(Gr.SphericalGrid.grdtospec(- PV.T.VerticalFlux[:,:,k] + PV.T.VerticalFlux[:,:,k-1]),dp_ratio)),
            #                      np.min(np.multiply(Gr.SphericalGrid.grdtospec(- PV.T.VerticalFlux[:,:,k] + PV.T.VerticalFlux[:,:,k-1]),dp_ratio)))
            # print('Thermal_expension',np.max(Thermal_expension),np.min(Thermal_expension))
            # print('forcing',np.max(PV.T.forcing[:,k]),np.min(PV.T.forcing[:,k]))
            # print("=====================================================================")

            # zero tendencies
            # PV.P.tendency[:,k]           = np.zeros_like(PV.P.spectral[:,k])
            # PV.Divergence.tendency[:,k]  = np.zeros_like(PV.Divergence.spectral[:,k])
            # PV.Vorticity.tendency[:,k]   = np.zeros_like(PV.Vorticity.spectral[:,k])
            # PV.T.tendency[:,k]           = np.zeros_like(PV.T.spectral[:,k])
            PV.QT.tendency[:,k]          = np.zeros_like(PV.QT.spectral[:,k])


        # josef's code
        ps  = PV.P.values[:,:,3]

        ### level 3 ###
        vrt3 = PV.Vorticity.values[:,:,2]
        div3 = PV.Divergence.values[:,:,2]
        vrtsp3 = PV.Vorticity.spectral[:,2]
        divsp3 = PV.Divergence.spectral[:,2]
        u3,v3 = Gr.SphericalGrid.getuv(vrtsp3,divsp3)
        temp3  = PV.T.values[:,:,2]
        ### level 2 ###
        vrt2 = PV.Vorticity.values[:,:,1]
        div2 = PV.Divergence.values[:,:,1]
        vrtsp2 = PV.Vorticity.spectral[:,1]
        divsp2 = PV.Divergence.spectral[:,1]
        u2,v2 = Gr.SphericalGrid.getuv(vrtsp2,divsp2)
        temp2  = PV.T.values[:,:,1]
        ### level 1 ###
        vrt1 = PV.Vorticity.values[:,:,0]
        div1 = PV.Divergence.values[:,:,0]
        vrtsp1 = PV.Vorticity.spectral[:,0]
        divsp1 = PV.Divergence.spectral[:,0]
        u1,v1 = Gr.SphericalGrid.getuv(vrtsp1,divsp1)
        temp1  = PV.T.values[:,:,0]

        # pressures
        p1 = PV.P.values[:,:,0]
        p2 = PV.P.values[:,:,1]
        p3 = PV.P.values[:,:,2]
        ps = PV.P.values[:,:,3]

        omega2 = (p1-p2)*PV.Divergence.values[:,:,0]
        omega3 = omega2+(p2-p3)*PV.Divergence.values[:,:,1]
        omegas = omega3+(p3-ps)*PV.Divergence.values[:,:,2]

        omegasp2 = Gr.SphericalGrid.grdtospec(omega2)
        omegasp3 = Gr.SphericalGrid.grdtospec(omega3)
        omegasps = Gr.SphericalGrid.grdtospec(omegas)

        tmp1 = np.multiply(u3,(p3-ps))
        tmp2 = np.multiply(v3,(p3-ps))
        ps_vrt, ps_div = Gr.SphericalGrid.getvrtdivspec(tmp1, tmp2)

        #surface pressure tendency
        dpssp = ps_div + omegasp3

        bulku21=0.5*omega2*(u2-u1)/(p2-p1)
        bulkv21=0.5*omega2*(v2-v1)/(p2-p1)
        bulktemp21=0.5*omega2*(temp2+temp1)/(p2-p1) #flux form

        bulku32=0.5*omega3*(u3-u2)/(p3-p2)
        bulkv32=0.5*omega3*(v3-v2)/(p3-p2)
        bulktemp32=0.5*omega3*(temp3+temp2)/(p3-p2)

        bulku3s=0.5*omega3*(u3-u2)/(ps-p3)
        bulkv3s=0.5*omega3*(v3-v2)/(ps-p3)
        bulktemp3s=omegas*temp3/(ps-p3)

        vrtsp21, divsp21 = Gr.SphericalGrid.getvrtdivspec(bulku21, bulkv21)
        vrtsp32, divsp32 = Gr.SphericalGrid.getvrtdivspec(bulku32, bulkv32)
        vrtsp3s, divsp3s = Gr.SphericalGrid.getvrtdivspec(bulku3s, bulkv3s)


        # geopotentials
        phi3 = Gr.Rd * temp3*np.log(ps/p3)
        phi2 = Gr.Rd * temp2*np.log(p3/p2)  + phi3
        phi1 = Gr.Rd * temp1*np.log(p2/p1)  + phi2


        # kinetic energy
        ekin3=0.5*(u3**2+v3**2)
        ekin2=0.5*(u2**2+v2**2)
        ekin1=0.5*(u1**2+v1**2)

        ### level 3 ###
        tmp31 = u3*(vrt3+Gr.Coriolis)
        tmp32 = v3*(vrt3+Gr.Coriolis)
        tmpa3, tmpb3 = Gr.SphericalGrid.getvrtdivspec(tmp31, tmp32)
        # vorticity tendnecy
        dvrtsp3 = -tmpb3  -vrtsp3s + PV.Vorticity.forcing[:,2]
        tmp33 = u3*(temp3)
        tmp34 = v3*(temp3)
        tmpd3, tmpe3 = Gr.SphericalGrid.getvrtdivspec(tmp33,tmp34)
        # temperature tendnecy
        dtempsp3 = -tmpe3 -Gr.SphericalGrid.grdtospec(-omegas*phi3/(ps-p3)/Gr.cp +bulktemp3s -bulktemp32*(p3-p2)/(ps-p3)) + PV.T.forcing[:,2]
        tmpf3 = Gr.SphericalGrid.grdtospec(phi3+ekin3)
        # divergence tendnecy
        ddivsp3 = tmpa3 - Gr.SphericalGrid.lap*tmpf3 -divsp3s + PV.Divergence.forcing[:,2]

        ### level 2 ###
        tmp21 = u2*(vrt2+Gr.Coriolis)
        tmp22 = v2*(vrt2+Gr.Coriolis)
        tmpa2, tmpb2 = Gr.SphericalGrid.getvrtdivspec(tmp21, tmp22)
        # vorticity tendnecy
        dp_ratio = Gr.SphericalGrid.grdtospec((p2-p1)/(p3-p2))
        dvrtsp2 = -tmpb2  -vrtsp32  -vrtsp21*dp_ratio + PV.Vorticity.forcing[:,1] # YAIR 
        tmp23 = u2*(temp2)
        tmp24 = v2*(temp2)
        tmpd2, tmpe2 = Gr.SphericalGrid.getvrtdivspec(tmp23,tmp24)
        # temperature tendnecy
        dtempsp2 = -tmpe2 -Gr.SphericalGrid.grdtospec(omega3*(phi3-phi2)/(p3-p2)/Gr.cp +bulktemp32 -bulktemp21*(p2-p1)/(p3-p2)) + PV.T.forcing[:,1]
        tmpf2 = Gr.SphericalGrid.grdtospec(phi2+ekin2)
        # divergence tendnecy
        ddivsp2 = tmpa2 - Gr.SphericalGrid.lap*tmpf2 - divsp32 -divsp21*dp_ratio + PV.Divergence.forcing[:,1] # YAIR

        ### level 1 ###
        tmp11 = u1*(vrt1+Gr.Coriolis)
        tmp12 = v1*(vrt1+Gr.Coriolis)
        tmpa1, tmpb1 = Gr.SphericalGrid.getvrtdivspec(tmp11, tmp12)
        # vorticity tendnecy
        dvrtsp1 = - tmpb1 - vrtsp21 + PV.Vorticity.forcing[:,0]
        tmp13 = u1*(temp1)
        tmp14 = v1*(temp1)
        tmpd1, tmpe1 = Gr.SphericalGrid.getvrtdivspec(tmp13,tmp14)
        # temperature tendnecy
        dtempsp1 = -tmpe1 -Gr.SphericalGrid.grdtospec(omega2*(phi2-phi1)/(p2-p1)/Gr.cp +bulktemp21) + PV.T.forcing[:,0]
        tmpf1 = Gr.SphericalGrid.grdtospec(phi1+ekin1)
        # divergence tendnecy
        ddivsp1 = tmpa1 -Gr.SphericalGrid.lap*tmpf1 -divsp21  + PV.Divergence.forcing[:,0]

        # print('=======================================')
        # print(np.max(np.abs(PV.Divergence.tendency[:,0] - ddivsp1)))
        # print(np.max(np.abs(PV.Divergence.tendency[:,1] - ddivsp2)))
        # print(np.max(np.abs(PV.Divergence.tendency[:,2] - ddivsp3)))

        # print(np.max(np.abs(PV.Vorticity.tendency[:,0] - ddivsp1)))
        # print(np.max(np.abs(PV.Vorticity.tendency[:,1] - ddivsp2)))
        # print(np.max(np.abs(PV.Vorticity.tendency[:,2] - ddivsp3)))

        # print(np.max(np.abs(PV.T.tendency[:,0] - dtempsp1)))
        # print(np.max(np.abs(PV.T.tendency[:,1] - dtempsp2)))
        # print(np.max(np.abs(PV.T.tendency[:,2] - dtempsp3)))

        # print(np.max(np.abs(PV.P.tendency[:,3] - dpssp)))
        # print('=======================================')
        # plt.figure()
        # plt.show()

        PV.Divergence.tendency[:,0] = ddivsp1
        PV.Vorticity.tendency[:,0] = dvrtsp1
        PV.T.tendency[:,0] = dtempsp1

        PV.Divergence.tendency[:,1] = ddivsp2
        PV.Vorticity.tendency[:,1] = dvrtsp2
        PV.T.tendency[:,1] = dtempsp2

        PV.Divergence.tendency[:,2] = ddivsp3
        PV.Vorticity.tendency[:,2] = dvrtsp3
        PV.T.tendency[:,2] = dtempsp3

        PV.P.tendency[:,3] = dpssp

        return