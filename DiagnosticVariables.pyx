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
    void diagnostic_variables(double Rd, double Rv,
           double* p,  double* T,  double* qt, double* ql, double* u,
           double* v, double* div, double* ke, double* wp, double* gz,
           Py_ssize_t k, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

cdef class DiagnosticVariable:
    def __init__(self, nx,ny, nl , kind, name, units):
        self.kind = kind
        self.name = name
        self.units = units
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        if name=='u' or name=='v':
            self.SurfaceFlux =  np.zeros((nx,ny),dtype=np.float64, order='c')
            self.forcing =  np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        return

cdef class DiagnosticVariables:
    def __init__(self, Parameters Pr, Grid Gr):
        self.KE = DiagnosticVariable(Gr.nx, Gr.ny, Gr.nl,   'kinetic_enetry',      'Ek','m^2/s^2' )
        self.QL = DiagnosticVariable(Gr.nx, Gr.ny, Gr.nl,   'liquid water specific humidity', 'ql','kg/kg' )
        self.gZ = DiagnosticVariable(Gr.nx, Gr.ny, Gr.nl+1, 'Geopotential', 'z','m^/s^2' )
        self.Wp = DiagnosticVariable(Gr.nx, Gr.ny, Gr.nl+1, 'Wp', 'w','pasc/s' )
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t k, j
        # self.VT.values.base     = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        # self.TT.values.base     = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        # self.UV.values.base     = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        for k in range(Pr.n_layers):
            j = Pr.n_layers-k-1 # geopotential is computed bottom -> up
            self.gZ.values.base[:,:,j] = np.add(np.multiply(np.multiply(Pr.Rd,PV.T.values[:,:,j]),np.log(np.divide(PV.P.values[:,:,j+1],PV.P.values[:,:,j]))),self.gZ.values[:,:,j+1])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_QL')
        Stats.add_zonal_mean('zonal_mean_QL')
        Stats.add_meridional_mean('meridional_mean_QL')
        Stats.add_global_mean('global_mean_KE')
        Stats.add_global_mean('global_mean_gZ')
        Stats.add_zonal_mean('zonal_mean_gZ')
        Stats.add_zonal_mean('zonal_mean_Wp')
        Stats.add_meridional_mean('meridional_mean_gZ')
        Stats.add_meridional_mean('meridional_mean_Wp')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_QL', self.QL.values)
        Stats.write_zonal_mean('zonal_mean_QL',self.QL.values)
        Stats.write_meridional_mean('meridional_mean_QL',self.QL.values)
        Stats.write_global_mean('global_mean_KE', self.KE.values)
        Stats.write_global_mean('global_mean_gZ', self.gZ.values[:,:,0:3])
        Stats.write_zonal_mean('zonal_mean_Wp',self.Wp.values[:,:,1:4])
        Stats.write_zonal_mean('zonal_mean_gZ',self.gZ.values[:,:,0:3])
        Stats.write_meridional_mean('meridional_mean_Wp',self.Wp.values[:,:,1:4])
        Stats.write_meridional_mean('meridional_mean_gZ',self.gZ.values[:,:,0:3])
        return

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Pr, Gr, int(TS.t), Pr.n_layers, 'Geopotential',  self.gZ.values[:,:,0:Pr.n_layers])
        Stats.write_3D_variable(Pr, Gr, int(TS.t), Pr.n_layers, 'Wp',            self.Wp.values[:,:,1:Pr.n_layers+1])
        Stats.write_3D_variable(Pr, Gr, int(TS.t), Pr.n_layers, 'QL',            self.QL.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t), Pr.n_layers, 'Kinetic_enegry',self.KE.values)
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t i, j, k
            Py_ssize_t ii = 1
            Py_ssize_t nx = Gr.nx
            Py_ssize_t ny = Gr.ny
            Py_ssize_t nl = Gr.nl
            double dxi = 1.0/Pr.dx
            double dyi = 1.0/Pr.dy
            double Rm

        self.Wp.values.base[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values.base[:,:,nl] = np.zeros_like(self.Wp.values[:,:,0])
        with nogil:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nl):
                        self.Divergence.values[i,j,k] = ((PV.U.values[i+1,j,k]-PV.U.values[i-1,j,k])*dxi +
                                                       (PV.V.values[i,j+1,k]-PV.V.values[i,j-1,k])*dyi)

                diagnostic_variables(Pr.Rd, Pr.Rv, &PV.P.values[0,0,0], &PV.T.values[0,0,0],
                                     &PV.QT.values[0,0,0],   &self.QL.values[0,0,0], &PV.U.values[0,0,0],
                                     &PV.V.values[0,0,0],  &self.Divergence.values[0,0,0],&self.KE.values[0,0,0],
                                     &self.Wp.values[0,0,0], &self.gZ.values[0,0,0], k, nx, ny, nl)
        return
