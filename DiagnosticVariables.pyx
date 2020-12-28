import cython
import numpy as np
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping
from Parameters cimport Parameters

cdef class DiagnosticVariable:
    def __init__(self, nx,ny,nl,n_spec, kind, name, units):
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.kind = kind
        self.name = name
        self.units = units
        return

cdef class DiagnosticVariables:
    def __init__(self, Parameters Pr, Grid Gr):
        self.U  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'zonal_velocity'     , 'u','m/s' )
        self.V  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'meridional_velocity', 'v','m/s' )
        self.KE = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'kinetic_enetry',      'Ek','m^2/s^2' )
        self.gZ = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Geopotential', 'z','m^/s^2' )
        self.Wp = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Wp', 'w','pasc/s' )
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t k, j
        for k in range(Pr.n_layers):
            self.U.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.U.values.base[:,:,k])
            self.V.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.V.values.base[:,:,k])
            self.KE.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.KE.values.base[:,:,k])
            self.gZ.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.gZ.values.base[:,:,k])
            self.Wp.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Wp.values.base[:,:,k])
            j = Pr.n_layers-k-1 # geopotential is computed bottom -> up
            self.gZ.values.base[:,:,j]  = np.multiply(Pr.Rd,np.multiply(PV.T.values[:,:,j],np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j])))) + self.gZ.values[:,:,j+1]
            self.gZ.spectral.base[:,j] = Gr.SphericalGrid.grdtospec(self.gZ.values.base[:,:,j])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
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

    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
        for k in range(Pr.n_layers):
            self.U.spectral.base[:,k]    = Gr.SphericalGrid.grdtospec(self.U.values.base[:,:,k])
            self.V.spectral.base[:,k]    = Gr.SphericalGrid.grdtospec(self.V.values.base[:,:,k])
            self.Wp.spectral.base[:,k+1] = Gr.SphericalGrid.grdtospec(self.Wp.values.base[:,:,k+1])
            self.gZ.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(self.gZ.values.base[:,:,k])
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_KE', self.KE.values)
        Stats.write_global_mean('global_mean_gZ', self.gZ.values[:,:,0:3])
        Stats.write_zonal_mean('zonal_mean_U',self.U.values)
        Stats.write_zonal_mean('zonal_mean_V',self.V.values)
        Stats.write_zonal_mean('zonal_mean_Wp',self.Wp.values[:,:,1:4])
        Stats.write_zonal_mean('zonal_mean_gZ',self.gZ.values[:,:,0:3])
        Stats.write_meridional_mean('meridional_mean_U',self.U.values)
        Stats.write_meridional_mean('meridional_mean_V',self.V.values)
        Stats.write_meridional_mean('meridional_mean_Wp',self.Wp.values[:,:,1:4])
        Stats.write_meridional_mean('meridional_mean_gZ',self.gZ.values[:,:,0:3])
        return

    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'Geopotential',  self.gZ.values[:,:,0:Gr.n_layers])
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'Wp',            self.Wp.values[:,:,1:Gr.n_layers+1])
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'U',             self.V.values)
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'V',             self.V.values)
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'Kinetic_enegry',self.KE.values)
        return

    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t j, k

        self.Wp.values.base[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values.base[:,:,Pr.n_layers] = np.zeros_like(self.Wp.values[:,:,0])
        for k in range(Pr.n_layers): # n_layers = 3; k=0,1,2
            j = Pr.n_layers-k-1 # geopotential is computed bottom -> up
            self.U.values.base[:,:,k], self.V.values.base[:,:,k] = Gr.SphericalGrid.getuv(
                         PV.Vorticity.spectral.base[:,k],PV.Divergence.spectral.base[:,k])
            self.KE.values.base[:,:,k]    = 0.5*np.add(np.power(self.U.values[:,:,k],2.0),np.power(self.V.values[:,:,k],2.0))
            self.Wp.values.base[:,:,k+1]  = np.subtract(self.Wp.values[:,:,k],
                                       np.multiply(np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k]),PV.Divergence.values[:,:,k]))
            self.gZ.values.base[:,:,j]  = np.add(np.multiply(np.multiply(Pr.Rd,PV.T.values[:,:,j]),np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))),self.gZ.values[:,:,j+1])
        return

    # # yair - need to add here diagnostic functions of stats
