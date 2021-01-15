import cython
import numpy as np
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping
from Parameters cimport Parameters
import time

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
        self.QL = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'liquid water specific humidity', 'ql','kg/kg' )
        self.gZ = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Geopotential', 'z','m^/s^2' )
        self.Wp = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers+1, Gr.SphericalGrid.nlm, 'Wp', 'w','pasc/s' )
        self.VT = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'temperature_flux', 'vT','Km/s' )
        self.TT = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'temperature_variance', 'TT','K^2' )
        self.UV = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'momentum_flux', 'uv','m^2/s^2' )
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t k, j
        self.VT.values.base     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.TT.values.base     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.UV.values.base     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        for k in range(Pr.n_layers):
            self.U.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.U.values.base[:,:,k])
            self.V.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.V.values.base[:,:,k])
            self.KE.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.KE.values.base[:,:,k])
            self.QL.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.gZ.values.base[:,:,k])
            self.Wp.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Wp.values.base[:,:,k])
            self.VT.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(np.multiply(self.V.values[:,:,k],PV.T.values[:,:,k]))
            self.TT.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(np.multiply(PV.T.values[:,:,k],PV.T.values[:,:,k]))
            self.UV.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(np.multiply(self.V.values[:,:,k],self.U.values[:,:,k]))
            j = Pr.n_layers-k-1 # geopotential is computed bottom -> up
            self.gZ.values.base[:,:,j] = np.add(np.multiply(np.multiply(Pr.Rd,PV.T.values[:,:,j]),np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))),self.gZ.values[:,:,j+1])
            self.gZ.spectral.base[:,j] = Gr.SphericalGrid.grdtospec(self.gZ.values.base[:,:,j])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_QL')
        Stats.add_zonal_mean('zonal_mean_QL')
        Stats.add_meridional_mean('meridional_mean_QL')
        Stats.add_global_mean('global_mean_KE')
        Stats.add_global_mean('global_mean_gZ')
        Stats.add_zonal_mean('zonal_mean_U')
        Stats.add_zonal_mean('zonal_mean_V')
        Stats.add_zonal_mean('zonal_mean_gZ')
        Stats.add_zonal_mean('zonal_mean_Wp')
        Stats.add_zonal_mean('zonal_mean_VT')
        Stats.add_zonal_mean('zonal_mean_TT')
        Stats.add_zonal_mean('zonal_mean_UV')
        Stats.add_meridional_mean('meridional_mean_U')
        Stats.add_meridional_mean('meridional_mean_V')
        Stats.add_meridional_mean('meridional_mean_gZ')
        Stats.add_meridional_mean('meridional_mean_Wp')
        Stats.add_meridional_mean('meridional_mean_VT')
        Stats.add_meridional_mean('meridional_mean_TT')
        Stats.add_meridional_mean('meridional_mean_UV')
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
        for k in range(Pr.n_layers):
            self.U.spectral.base[:,k]    = Gr.SphericalGrid.grdtospec(self.U.values.base[:,:,k])
            self.V.spectral.base[:,k]    = Gr.SphericalGrid.grdtospec(self.V.values.base[:,:,k])
            self.Wp.spectral.base[:,k+1] = Gr.SphericalGrid.grdtospec(self.Wp.values.base[:,:,k+1])
            self.gZ.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(self.gZ.values.base[:,:,k])
            self.VT.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(self.VT.values.base[:,:,k])
            self.TT.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(self.TT.values.base[:,:,k])
            self.UV.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(self.UV.values.base[:,:,k])
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_QL', self.QL.values)
        Stats.write_zonal_mean('zonal_mean_QL',self.QL.values)
        Stats.write_meridional_mean('meridional_mean_QL',self.QL.values)
        Stats.write_global_mean('global_mean_KE', self.KE.values)
        Stats.write_global_mean('global_mean_gZ', self.gZ.values[:,:,0:3])
        Stats.write_zonal_mean('zonal_mean_U',self.U.values)
        Stats.write_zonal_mean('zonal_mean_V',self.V.values)
        Stats.write_zonal_mean('zonal_mean_Wp',self.Wp.values[:,:,1:4])
        Stats.write_zonal_mean('zonal_mean_gZ',self.gZ.values[:,:,0:3])
        Stats.write_zonal_mean('zonal_mean_UV',self.UV.values.base)
        Stats.write_zonal_mean('zonal_mean_VT',self.VT.values.base)
        Stats.write_zonal_mean('zonal_mean_TT',self.TT.values.base)
        Stats.write_meridional_mean('meridional_mean_U',self.U.values)
        Stats.write_meridional_mean('meridional_mean_V',self.V.values)
        Stats.write_meridional_mean('meridional_mean_Wp',self.Wp.values[:,:,1:4])
        Stats.write_meridional_mean('meridional_mean_gZ',self.gZ.values[:,:,0:3])
        Stats.write_meridional_mean('meridional_mean_UV',self.UV.values)
        Stats.write_meridional_mean('meridional_mean_VT',self.VT.values)
        Stats.write_meridional_mean('meridional_mean_TT',self.TT.values)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Geopotential',  self.gZ.values[:,:,0:Pr.n_layers])
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Wp',            self.Wp.values[:,:,1:Pr.n_layers+1])
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'QL',            self.QL.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'U',             self.U.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'V',             self.V.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Kinetic_enegry',self.KE.values)
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t j, k
            Py_ssize_t ii = 1
            Py_ssize_t nl = Pr.n_layers

        self.Wp.values.base[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values.base[:,:,nl] = np.zeros_like(self.Wp.values[:,:,0])
        for k in range(nl):
            j = nl-k-ii # geopotential is computed bottom -> up
            t0 = time.clock()
            self.U.values.base[:,:,k], self.V.values.base[:,:,k] = Gr.SphericalGrid.getuv(
                         PV.Vorticity.spectral.base[:,k],PV.Divergence.spectral.base[:,k])
            self.KE.values.base[:,:,k]    = np.multiply(0.5,np.add(np.power(self.U.values[:,:,k],2.0),
                                                                   np.power(self.V.values[:,:,k],2.0)))
            self.Wp.values.base[:,:,k+1]  = np.subtract(self.Wp.values[:,:,k],
                                       np.multiply(np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k]),PV.Divergence.values[:,:,k]))
            Rm = np.add(np.multiply(Pr.Rd,np.subtract(1.0,PV.QT.values[:,:,j])),np.multiply(Pr.Rv,np.subtract(PV.QT.values[:,:,j],self.QL.values[:,:,j])))
            self.gZ.values.base[:,:,j]  = np.add(np.multiply(np.multiply(Pr.Rd,PV.T.values[:,:,j]),np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))),self.gZ.values[:,:,j+1])

            self.VT.values.base[:,:,k] = np.multiply(self.V.values[:,:,k],PV.T.values[:,:,k])
            self.TT.values.base[:,:,k] = np.multiply(PV.T.values[:,:,k],PV.T.values[:,:,k])
            self.UV.values.base[:,:,k] = np.multiply(self.V.values[:,:,k],self.U.values[:,:,k])
        return
