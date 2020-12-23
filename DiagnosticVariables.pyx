import cython
import numpy as np
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping

cdef class DiagnosticVariable:
    def __init__(self, nx,ny,nl,n_spec, kind, name, units):
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.kind = kind
        self.name = name
        self.units = units
        return

cdef class DiagnosticVariables:
    def __init__(self, Grid Gr):
        self.U  = DiagnosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,   Gr.SphericalGrid.nlm, 'zonal_velocity'     , 'u','m/s' )
        self.V  = DiagnosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,   Gr.SphericalGrid.nlm, 'meridional_velocity', 'v','m/s' )
        self.KE = DiagnosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,   Gr.SphericalGrid.nlm, 'kinetic_enetry',      'Ek','m^2/s^2' )
        self.gZ = DiagnosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers+1, Gr.SphericalGrid.nlm, 'Geopotential', 'z','m^/s^2' )
        self.Wp = DiagnosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers+1, Gr.SphericalGrid.nlm, 'Wp', 'w','pasc/s' )
        return

    cpdef initialize(self, Grid Gr):
        cdef:
            Py_ssize_t k
        self.U.values  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),    dtype=np.float64, order='c')
        self.V.values  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),    dtype=np.float64, order='c')
        self.KE.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),    dtype=np.float64, order='c')
        self.gZ.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers+1),  dtype=np.float64, order='c')
        self.Wp.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers+1),  dtype=np.float64, order='c')
        for k in range(Gr.n_layers):
            U = Gr.SphericalGrid.grdtospec(np.float64(self.U.values[:,:,k]))
            V = Gr.SphericalGrid.grdtospec(np.float64(self.V.values[:,:,k]))
            KE = Gr.SphericalGrid.grdtospec(np.float64(self.KE.values[:,:,k]))
            gZ = Gr.SphericalGrid.grdtospec(np.float64(self.gZ.values[:,:,k]))
            Wp = Gr.SphericalGrid.grdtospec(np.float64(self.Wp.values[:,:,k]))
            for i  in range(len(U)):
                self.U.spectral[i,k]  = U[i]
                self.V.spectral[i,k]  = V[i]
                self.KE.spectral[i,k] = KE[i]
                self.gZ.spectral[i,k] = gZ[i]
                self.Wp.spectral[i,k] = Wp[i]
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

    cpdef physical_to_spectral(self, Grid Gr):
        cdef:
            Py_ssize_t k
        for k in range(Gr.n_layers):
            self.U.spectral[:,k]    = np.array(Gr.SphericalGrid.grdtospec(self.U.values[:,:,k]))
            self.V.spectral[:,k]    = np.array(Gr.SphericalGrid.grdtospec(self.V.values[:,:,k]))
            self.Wp.spectral[:,k+1] = np.array(Gr.SphericalGrid.grdtospec(self.Wp.values[:,:,k+1]))
            self.gZ.spectral[:,k]   = np.array(Gr.SphericalGrid.grdtospec(self.gZ.values[:,:,k]))
        return

    # # convert spectral data to spherical
    # # I need to define this function to ast on a general variable
    # # cpdef spectral_to_physical(self):
    # #     cdef:
    # #         Py_ssize_t k
    # #     for k in range(self.n_layers):
    # #         self.U.values[:,:,k], self.V.value[:,:,k] = np.array(Gr.SphericalGrid.getuv(self.Vorticity.spectral[:,k], self.Divergence.spectral[:,k]))
    # #         self.Wp.values[:,:,k+1] = np.array(Gr.SphericalGrid.spectogrd(self.Wp.spectral[:,k+1]))
    # #         self.gZ.values[:,:,k] = np.array(Gr.SphericalGrid.spectogrd(self.gZ.spectral[:,k]))
    # #     return

    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
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

    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'Geopotential',  self.gZ.values[:,:,0:Gr.n_layers])
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'Wp',            self.Wp.values[:,:,1:Gr.n_layers+1])
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'U',             self.V.values)
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'V',             self.V.values)
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'Kinetic_enegry',self.KE.values)
        return

    cpdef update(self, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t j, k

        self.Wp.values[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values[:,:,Gr.n_layers] = np.zeros_like(self.Wp.values[:,:,0])
        for k in range(Gr.n_layers): # n_layers = 3; k=0,1,2
            j = Gr.n_layers-k-1 # geopotential is computed bottom -> up
            U, V = Gr.SphericalGrid.getuv(PV.Vorticity.spectral[:,k],PV.Divergence.spectral[:,k])
            self.U.values[:,:,k] = np.array(U)
            self.V.values[:,:,k] = np.array(V)
            self.KE.values[:,:,k]    = 0.5*np.add(np.power(U,2.0),np.power(V,2.0))
            self.Wp.values[:,:,k+1]  = np.subtract(self.Wp.values[:,:,k],
                                       np.multiply(np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k]),PV.Divergence.values[:,:,k]))
            self.gZ.values[:,:,j]  = np.add(np.multiply(np.multiply(Gr.Rd,PV.T.values[:,:,j]),np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))),self.gZ.values[:,:,j+1])
        return

    # # yair - need to add here diagnostic functions of stats
