import numpy as np
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
from Parameters import Parameters

class DiagnosticVariable:
    def __init__(self, nx,ny,nl,n_spec, kind, name, units):
        self.kind = kind
        self.name = name
        self.units = units
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        # self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        if name=='u' or name=='v':
            self.SurfaceFlux =  np.zeros((nx,ny),dtype=np.float64, order='c')
            self.forcing =  np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        return

class DiagnosticVariables:
    def __init__(self, Pr, Gr):
        self.U  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'zonal_velocity'     , 'u','m/s' )
        self.V  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'meridional_velocity', 'v','m/s' )
        self.KE = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'kinetic_enetry',      'Ek','m^2/s^2' )
        self.QL = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'liquid water specific humidity', 'ql','kg/kg' )
        self.gZ = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Geopotential', 'z','m^/s^2' )
        self.Wp = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Wp', 'w','pasc/s' )
        self.VT = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'temperature_flux', 'vT','Km/s' )
        self.TT = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'temperature_variance', 'TT','K^2' )
        self.UV = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'momentum_flux', 'uv','m^2/s^2' )
        self.M  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'angular_momentum', 'M','m^2/s' )
        return

    def initialize_io(self, Pr, Stats):
        if Pr.moist_index > 0.0:
            Stats.add_global_mean('global_mean_QL')
            Stats.add_zonal_mean('zonal_mean_QL')
            Stats.add_meridional_mean('meridional_mean_QL')
        Stats.add_global_mean('global_mean_KE')
        Stats.add_global_mean('global_mean_gZ')
        Stats.add_global_mean('global_mean_M')
        Stats.add_zonal_mean('zonal_mean_U')
        Stats.add_zonal_mean('zonal_mean_V')
        Stats.add_zonal_mean('zonal_mean_gZ')
        Stats.add_zonal_mean('zonal_mean_Wp')
        Stats.add_zonal_mean('zonal_mean_VT')
        Stats.add_zonal_mean('zonal_mean_TT')
        Stats.add_zonal_mean('zonal_mean_UV')
        Stats.add_zonal_mean('zonal_mean_M')
        Stats.add_meridional_mean('meridional_mean_U')
        Stats.add_meridional_mean('meridional_mean_V')
        Stats.add_meridional_mean('meridional_mean_gZ')
        Stats.add_meridional_mean('meridional_mean_Wp')
        Stats.add_meridional_mean('meridional_mean_VT')
        Stats.add_meridional_mean('meridional_mean_TT')
        Stats.add_meridional_mean('meridional_mean_UV')
        Stats.add_meridional_mean('meridional_mean_M')
        return

    def stats_io(self, Gr, Pr, Stats):
        if Pr.moist_index > 0.0:
            Stats.write_global_mean(Gr, 'global_mean_QL', self.QL.values)
            Stats.write_zonal_mean('zonal_mean_QL',self.QL.values)
            Stats.write_meridional_mean('meridional_mean_QL',self.QL.values)
        Stats.write_global_mean(Gr, 'global_mean_KE', self.KE.values)
        Stats.write_global_mean(Gr, 'global_mean_gZ', self.gZ.values[:,:,0:-1])
        Stats.write_global_mean(Gr, 'global_mean_M', self.M.values)
        Stats.write_zonal_mean('zonal_mean_U',self.U.values)
        Stats.write_zonal_mean('zonal_mean_V',self.V.values)
        Stats.write_zonal_mean('zonal_mean_Wp',self.Wp.values[:,:,1:])
        Stats.write_zonal_mean('zonal_mean_gZ',self.gZ.values[:,:,0:-1])
        Stats.write_zonal_mean('zonal_mean_UV',self.UV.values)
        Stats.write_zonal_mean('zonal_mean_VT',self.VT.values)
        Stats.write_zonal_mean('zonal_mean_TT',self.TT.values)
        Stats.write_zonal_mean('zonal_mean_M',self.M.values)
        Stats.write_meridional_mean('meridional_mean_U',self.U.values)
        Stats.write_meridional_mean('meridional_mean_V',self.V.values)
        Stats.write_meridional_mean('meridional_mean_Wp',self.Wp.values[:,:,1:])
        Stats.write_meridional_mean('meridional_mean_gZ',self.gZ.values[:,:,0:-1])
        Stats.write_meridional_mean('meridional_mean_UV',self.UV.values)
        Stats.write_meridional_mean('meridional_mean_VT',self.VT.values)
        Stats.write_meridional_mean('meridional_mean_TT',self.TT.values)
        Stats.write_meridional_mean('meridional_mean_M',self.TT.values)
        return

    def io(self, Pr, TS, Stats):
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Geopotential',  self.gZ.values[:,:,0:Pr.n_layers])
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Wp',            self.Wp.values[:,:,1:Pr.n_layers+1])
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'QL',            self.QL.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'U',             self.U.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'V',             self.V.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Kinetic_enegry',self.KE.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'angular_momentum',self.M.values)
        return

    def update(self, Pr, Gr, PV):
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers
        Rm = np.zeros_like(self.QL.values[:,:,0])

        self.TT.values = np.multiply(PV.T.values, PV.T.values)
        self.VT.values = np.multiply(self.V.values,PV.T.values)
        self.UV.values = np.multiply(self.V.values,self.U.values)
        self.Wp.values[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values[:,:,nl] = np.zeros_like(self.Wp.values[:,:,0])
        for k in range(nl):
            self.M.values[:,:,k]  = Pr.rsphere*np.cos(Gr.lat)*(Pr.rsphere*Pr.Omega*np.cos(Gr.lat) + self.U.values[:,:,k])
            self.U.values[:,:,k], self.V.values[:,:,k] = Gr.SphericalGrid.getuv(
                         PV.Vorticity.spectral[:,k],PV.Divergence.spectral[:,k])
            self.KE.values[:,:,k] = 0.5*np.add(np.power(self.U.values[:,:,k],2),np.power(self.V.values[:,:,k],2))
            self.Wp.values[:,:,k+1]  = np.subtract(self.Wp.values[:,:,k], np.multiply(np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k]),PV.Divergence.values[:,:,k]))
            k_rev = Pr.n_layers-k-1

            Rm[:,:] = np.add(np.multiply(Pr.Rd,np.subtract(1.0,PV.QT.values[:,:,k_rev])),
                        np.multiply(Pr.Rv,np.subtract(PV.QT.values[:,:,k_rev],self.QL.values[:,:,k_rev])))
            self.gZ.values[:,:,k_rev] = np.add(np.multiply(Rm[:,:],np.multiply(PV.T.values[:,:,k_rev],np.log(PV.P.values[:,:,k_rev+1]/PV.P.values[:,:,k_rev]))), self.gZ.values[:,:,k_rev+1])
            # for i in range(nx):
            #     for j in range(ny):
            #         Rm = Pr.Rd*(1.0-PV.QT.values[i,j,k_rev]) + Pr.Rv*(PV.QT.values[i,j,k_rev] - self.QL.values[i,j,k_rev])
            #         self.gZ.values[i,j,k_rev] = Rm*PV.T.values[i,j,k_rev]*np.log(PV.P.values[i,j,k_rev+1]/PV.P.values[i,j,k_rev]) + self.gZ.values[i,j,k_rev+1]
        return
