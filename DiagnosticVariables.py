import numpy as np
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping
from Parameters import Parameters

# def extern from "diagnostic_variables.h":
#     void diagnostic_variables(double Rd, double Rv, double Omega, double a,
#            double* lat,  double* p,  double* T,  double* qt, double* ql, double* u,  double* v,
#            double* div, double* ke, double* wp, double* gz, double* uv, double* TT, double* vT,
#            double* M, Py_ssize_t k, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

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
        Stats.write_zonal_mean('zonal_mean_UV',self.UV.values.base)
        Stats.write_zonal_mean('zonal_mean_VT',self.VT.values.base)
        Stats.write_zonal_mean('zonal_mean_TT',self.TT.values.base)
        Stats.write_zonal_mean('zonal_mean_M',self.M.values.base)
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
        ii = 1
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers

        self.Wp.values.base[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values.base[:,:,nl] = np.zeros_like(self.Wp.values[:,:,0])
        for k in range(nl):
            self.U.values.base[:,:,k], self.V.values.base[:,:,k] = Gr.SphericalGrid.getuv(
                         PV.Vorticity.spectral.base[:,k],PV.Divergence.spectral.base[:,k])
            diagnostic_variables(Pr.Rd, Pr.Rv, Pr.Omega, Pr.rsphere, &Gr.lat[0,0], &PV.P.values[0,0,0], &PV.T.values[0,0,0],
                                    &PV.QT.values[0,0,0],   &self.QL.values[0,0,0], &self.U.values[0,0,0],
                                    &self.V.values[0,0,0],  &PV.Divergence.values[0,0,0],
                                    &self.KE.values[0,0,0], &self.Wp.values[0,0,0], &self.gZ.values[0,0,0],
                                    &self.UV.values[0,0,0], &self.TT.values[0,0,0], &self.VT.values[0,0,0],
                                    &self.M.values[0,0,0], k, nx, ny, nl)
        return
