#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping
from Parameters cimport Parameters

cdef extern from "diagnostic_variables.h":
    void diagnostic_variables(double g, double* rho, double* gH_k, double* rho_gH_k, double* h, double* qt, double* ql,
           double* u, double* v, double* div, double* ke, double* uv, double* hh,
           double* vh, Py_ssize_t k, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil


cdef class DiagnosticVariable:
    def __init__(self, nx,ny,nl,n_spec, kind, name, units):
        self.kind = kind
        self.name = name
        self.units = units
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        if name=='u' or name=='v':
            self.SurfaceFlux =  np.zeros((nx,ny),dtype=np.float64, order='c')
            self.forcing =  np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        return

cdef class DiagnosticVariables:
    def __init__(self, Parameters Pr, Grid Gr):
        self.gH = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'geopotential'     , 'gH','m/s' )
        self.U  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'zonal_velocity'     , 'u','m/s' )
        self.V  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'meridional_velocity', 'v','m/s' )
        self.KE = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'kinetic_enetry',      'Ek','m^2/s^2' )
        self.QL = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'liquid water specific humidity', 'ql','kg/kg' )
        self.P  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'Pressure', 'p','pasc' )
        self.VH = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'Depth_flux', 'vH','m^2/s' )
        self.HH = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'Depth_variance', 'HH','m^2' )
        self.UV = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers, Gr.SphericalGrid.nlm, 'momentum_flux', 'uv','m^2/s^2' )
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t k, j
        for k in range(Pr.n_layers):
            self.U.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.U.values.base[:,:,k])
            self.V.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.V.values.base[:,:,k])
            self.KE.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.KE.values.base[:,:,k])
            self.QL.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.QL.values.base[:,:,k])
            self.VH.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(np.multiply(self.V.values[:,:,k],PV.H.values[:,:,k]))
            self.HH.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(np.multiply(PV.H.values[:,:,k],PV.H.values[:,:,k]))
            self.UV.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(np.multiply(self.V.values[:,:,k],self.U.values[:,:,k]))
            self.P.values.base[:,:,k] = np.sum(PV.H.values[:,:,0:k],axis = 2)
            self.P.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,k])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_U')
        Stats.add_global_mean('global_mean_V')
        Stats.add_global_mean('global_mean_KE')
        Stats.add_global_mean('global_mean_QL')
        Stats.add_global_mean('global_mean_VH')
        Stats.add_global_mean('global_mean_HH')
        Stats.add_global_mean('global_mean_UV')
        Stats.add_global_mean('global_mean_P')
        Stats.add_zonal_mean('zonal_mean_U')
        Stats.add_zonal_mean('zonal_mean_V')
        Stats.add_zonal_mean('zonal_mean_KE')
        Stats.add_zonal_mean('zonal_mean_QL')
        Stats.add_zonal_mean('zonal_mean_VH')
        Stats.add_zonal_mean('zonal_mean_HH')
        Stats.add_zonal_mean('zonal_mean_UV')
        Stats.add_zonal_mean('zonal_mean_P')
        Stats.add_meridional_mean('meridional_mean_U')
        Stats.add_meridional_mean('meridional_mean_V')
        Stats.add_meridional_mean('meridional_mean_KE')
        Stats.add_meridional_mean('meridional_mean_QL')
        Stats.add_meridional_mean('meridional_mean_VH')
        Stats.add_meridional_mean('meridional_mean_HH')
        Stats.add_meridional_mean('meridional_mean_UV')
        Stats.add_meridional_mean('meridional_mean_P')
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef physical_to_spectral(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t k
        for k in range(Pr.n_layers):
            self.U.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.U.values.base[:,:,k])
            self.V.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.V.values.base[:,:,k])
            self.P.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,k])
            self.KE.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.KE.values.base[:,:,k])
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_U', self.U.values)
        Stats.write_global_mean('global_mean_V', self.V.values)
        Stats.write_global_mean('global_mean_KE', self.KE.values)
        Stats.write_global_mean('global_mean_QL', self.QL.values)
        Stats.write_global_mean('global_mean_VH', self.VH.values)
        Stats.write_global_mean('global_mean_HH', self.HH.values)
        Stats.write_global_mean('global_mean_UV', self.UV.values)
        Stats.write_global_mean('global_mean_P', self.P.values)
        Stats.write_zonal_mean('zonal_mean_U',self.U.values)
        Stats.write_zonal_mean('zonal_mean_V',self.V.values)
        Stats.write_zonal_mean('zonal_mean_KE',self.KE.values)
        Stats.write_zonal_mean('zonal_mean_QL',self.QL.values)
        Stats.write_zonal_mean('zonal_mean_VH',self.VH.values)
        Stats.write_zonal_mean('zonal_mean_HH',self.HH.values)
        Stats.write_zonal_mean('zonal_mean_UV',self.UV.values)
        Stats.write_zonal_mean('zonal_mean_P',self.P.values)
        Stats.write_meridional_mean('meridional_mean_U',self.U.values)
        Stats.write_meridional_mean('meridional_mean_V',self.V.values)
        Stats.write_meridional_mean('meridional_mean_KE',self.KE.values)
        Stats.write_meridional_mean('meridional_mean_QL',self.QL.values)
        Stats.write_meridional_mean('meridional_mean_VH',self.VH.values)
        Stats.write_meridional_mean('meridional_mean_HH',self.HH.values)
        Stats.write_meridional_mean('meridional_mean_UV',self.UV.values)
        Stats.write_meridional_mean('meridional_mean_P',self.P.values)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Pressure',      self.P.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'QL',            self.QL.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'U',             self.U.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'V',             self.V.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Kinetic_enegry',self.KE.values)
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t k, k_rev
            Py_ssize_t ii = 1
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            double Rm
            double [:,:,:] gH_k     = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            double [:,:,:] rho_gH_k = np.zeros((nx,ny,nl),dtype=np.float64, order='c')

        for k in range(nl):
            self.U.values.base[:,:,k], self.V.values.base[:,:,k] = Gr.SphericalGrid.getuv(
                         PV.Vorticity.spectral.base[:,k],PV.Divergence.spectral.base[:,k])
            with nogil:
                diagnostic_variables(Pr.g, &Pr.rho[0], &gH_k[0,0,0], &rho_gH_k[0,0,0], &PV.H.values[0,0,0], &PV.QT.values[0,0,0],
                                     &self.QL.values[0,0,0], &self.U.values[0,0,0], &self.V.values[0,0,0],
                                     &PV.Divergence.values[0,0,0],&self.KE.values[0,0,0],
                                     &self.UV.values[0,0,0], &self.HH.values[0,0,0], &self.VH.values[0,0,0],
                                     k, nx, ny, nl)

            self.gH.values.base[:,:,k] = np.sum(gH_k[:,:,0:k],axis=2)
            self.P.values.base[:,:,k]  = np.sum(rho_gH_k[:,:,0:k],axis=2)
        return
