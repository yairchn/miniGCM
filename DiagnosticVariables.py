
import numpy as np
import matplotlib.pyplot as plt
import time
import NetCDFIO as Stats
from math import *


class DiagnosticVariable:
    def __init__(self, nx,ny,nl,n_spec, kind, name, units):
        self.values = np.zeros((nx,ny,nl),dtype=np.int64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.kind = kind
        self.name = name
        self.units = units
        return

class DiagnosticVariables:
    def __init__(self, Pr, Gr):
        self.U     = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'zonal_velocity'     , 'u','m/s' )
        self.V     = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'meridional_velocity', 'v','m/s' )
        self.KE    = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'kinetic_enetry',      'Ek','m^2/s^2' )
        self.gZ    = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Geopotential', 'z','m^/s^2' )
        self.Wp    = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Omega', 'Wp','pasc/s' )
        return

    def initialize(self, Pr, Gr, PV):
        self.U.values      = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.V.values      = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.KE.values     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.gZ.values     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers+1),  dtype=np.double, order='c') # josef place here the initial values for the variable 
        self.Wp.values     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers+1),  dtype=np.double, order='c') # josef place here the initial values for the variable 
        for k in range(Pr.n_layers):
            self.U.spectral[:,k]  = Gr.SphericalGrid.grdtospec(self.U.values[:,:,k])
            self.V.spectral[:,k]  = Gr.SphericalGrid.grdtospec(self.V.values[:,:,k])
            self.KE.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.KE.values[:,:,k])
            self.Wp.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.Wp.values[:,:,k])
            j = Pr.n_layers-k-1 # geopotential is computed bottom -> up
            self.gZ.values[:,:,j]  = np.multiply(Pr.Rd*PV.T.values[:,:,j],np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))) + self.gZ.values[:,:,j+1]
            self.gZ.spectral[:,j] = Gr.SphericalGrid.grdtospec(self.gZ.values[:,:,j])
        return

    def initialize_io(self, Stats):
        Stats.add_global_mean('global_mean_KE')
        Stats.add_global_mean('global_mean_gZ')
        Stats.add_zonal_mean('zonal_mean_U')
        Stats.add_zonal_mean('zonal_mean_V')
        Stats.add_zonal_mean('zonal_mean_gZ')
        Stats.add_zonal_mean('zonal_mean_Wp')
        Stats.add_meridional_mean('meridional_mean_U')
        Stats.add_meridional_mean('meridional_mean_V')
        Stats.add_meridional_mean('meridional_mean_gZ')
        Stats.add_meridional_mean('meridional_mean_Wp')
        return

    def physical_to_spectral(self, Pr, Gr):
        for k in range(Pr.n_layers):
            self.U.spectral[:,k]  = Gr.SphericalGrid.grdtospec(self.U.values[:,:,k])
            self.V.spectral[:,k]  = Gr.SphericalGrid.grdtospec(self.V.values[:,:,k])
            self.Wp.spectral[:,k+1] = Gr.SphericalGrid.grdtospec(self.Wp.values[:,:,k+1])
            self.gZ.spectral[:,k] = Gr.SphericalGrid.grdtospec(self.gZ.values[:,:,k])
        return

    # convert spectral data to spherical
    def spectral_to_physical(self):
        for k in range(self.n_layers):
            self.U.values[:,:,k], self.V.value[:,:,k] = Gr.SphericalGrid.getuv(self.Vorticity.spectral[:,k], self.Divergence.spectral[:,k])
            self.Wp.values[:,:,k+1] = Gr.SphericalGrid.spectogrd(self.Wp.spectral[:,k+1])
            self.gZ.values[:,:,k] = Gr.SphericalGrid.spectogrd(self.gZ.spectral[:,k])
        return

    def stats_io(self, TS, Stats):
        Stats.write_global_mean('global_mean_KE', self.KE.values, TS.t)
        Stats.write_global_mean('global_mean_gZ', self.gZ.values[:,:,0:3], TS.t)
        Stats.write_zonal_mean('zonal_mean_U',self.U.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_V',self.V.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_Wp',self.Wp.values[:,:,1:4], TS.t)
        Stats.write_zonal_mean('zonal_mean_gZ',self.gZ.values[:,:,0:3], TS.t)
        Stats.write_meridional_mean('meridional_mean_U',self.U.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_V',self.V.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_Wp',self.Wp.values[:,:,1:4], TS.t)
        Stats.write_meridional_mean('meridional_mean_gZ',self.gZ.values[:,:,0:3], TS.t)
        return

    def io(self, Pr, Gr, TS, Stats):
        Stats.write_3D_variable(Gr, int(TS.t), Pr.n_layers, self.gZ.name,self.gZ.values[:,:,0:Pr.n_layers])
        Stats.write_3D_variable(Gr, int(TS.t), Pr.n_layers, self.Wp.name,self.Wp.values[:,:,1:Pr.n_layers+1])
        Stats.write_3D_variable(Gr, int(TS.t), Pr.n_layers, self.V.name,self.V.values)
        Stats.write_3D_variable(Gr, int(TS.t), Pr.n_layers, self.V.name,self.V.values)
        Stats.write_3D_variable(Gr, int(TS.t), Pr.n_layers, self.KE.name,self.KE.values)
        return

    def update(self, Pr, Gr, PV):
        self.Wp.values[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values[:,:,Pr.n_layers] = np.zeros_like(self.Wp.values[:,:,0])
        for k in range(Pr.n_layers): # Pr.n_layers = 3; k=0,1,2
            j = Pr.n_layers-k-1 # geopotential is computed bottom -> up
            self.U.values[:,:,k], self.V.values[:,:,k] = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,k],PV.Divergence.spectral[:,k])
            self.KE.values[:,:,k]    = 0.5*np.add(np.power(self.U.values[:,:,k],2.0),np.power(self.V.values[:,:,k],2.0))
            self.Wp.values[:,:,k+1]  = self.Wp.values[:,:,k]-np.multiply(PV.P.values[:,:,k+1]-PV.P.values[:,:,k],PV.Divergence.values[:,:,k])
            self.gZ.values[:,:,j]  = np.multiply(Pr.Rd*PV.T.values[:,:,j],np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))) + self.gZ.values[:,:,j+1]
        return
    # yair - need to add here diagnostic functions of stats
