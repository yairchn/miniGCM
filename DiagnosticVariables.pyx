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
from libc.math cimport pow, log, fabs

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
        return

cdef class DiagnosticVariables:
    def __init__(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t ng = Gr.ng

        self.P          = DiagnosticVariable(Gr.nx+2*ng, Gr.ny+2*ng, Gr.nl,   'pressure',                       'p',   'kgm/s^2')
        self.Vel        = DiagnosticVariable(Gr.nx+2*ng, Gr.ny+2*ng, Gr.nl,   'windspeed',                      'Vel', 'm/s')
        self.KE         = DiagnosticVariable(Gr.nx+2*ng, Gr.ny+2*ng, Gr.nl,   'kinetic_enetry',                 'Ek',  'm^2/s^2')
        self.QL         = DiagnosticVariable(Gr.nx+2*ng, Gr.ny+2*ng, Gr.nl,   'liquid water specific humidity', 'ql',  'kg/kg')
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_QL')
        Stats.add_global_mean('global_mean_KE')
        Stats.add_axisymmetric_mean('axisymmetric_mean_QL')
        Stats.add_axisymmetric_mean('axisymmetric_mean_KE')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_QL', self.QL.values)
        Stats.write_global_mean('global_mean_KE', self.KE.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_QL',self.QL.values)
        Stats.write_axisymmetric_mean('axisymmetric_mean_KE',self.KE.values)
        return

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Pr, Gr, int(TS.t), Gr.nl, 'QL',            self.QL.values)
        Stats.write_3D_variable(Pr, Gr, int(TS.t), Gr.nl, 'Kinetic_enegry',self.KE.values)
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t i, j, k, k1
            Py_ssize_t ng = Gr.ng
            Py_ssize_t nx = Gr.nx
            Py_ssize_t ny = Gr.ny
            Py_ssize_t nl = Gr.nl
            double dxi = 1.0/Gr.dx
            double dyi = 1.0/Gr.dy
            double Rm
            double [:,:,:] H_tot = np.zeros((nx+2*ng,ny+2*ng,nl),dtype=np.float64, order='c')

        # ============| Note on pressure formualtion |============
        # in a model with i=1:n layers we define the sum of all underlying depthes (hᵢ) as:
        # Hᵢ = Σᵢⁿ(hᵢ);
        # the top layer pressure is: p₀ = gρ₀H₀
        # and for i>0 the pressure at the i'th layer is:
        # pᵢ = pᵢ₋₁ + g(ρᵢ-ρᵢ₋₁)Hᵢ

        # with nogil:
        for i in range(nx+2*ng):
            for j in range(ny+2*ng):
                for k in range(nl):
                    for kk in range(k,nl):
                        H_tot[i,j,k] += PV.H.values[i,j,kk]

        for k in range(nl):
            # with nogil:
            for i in range(nx+2*ng):
                for j in range(ny+2*ng):
                    self.Vel.values[i,j,k] = 0.5*(pow(PV.U.values[i,j,k]+PV.U.values[i,j,k],2.0)
                                                 +pow(PV.V.values[i,j,k]+PV.V.values[i,j,k],2.0))
                    self.KE.values.base[i,j,k] = 0.5*(pow((PV.U.values[i,j,k]+PV.U.values[i,j,k])/2.0,2.0)
                                                    + pow((PV.V.values[i,j,k]+PV.V.values[i,j,k])/2.0,2.0))
                    if k==0:
                        self.P.values.base[i,j,k] = Pr.rho[k]*Pr.g*H_tot[i,j,k]
                    if k>0:
                        self.P.values.base[i,j,k] = (self.P.values.base[i,j,k-1]
                                                    + Pr.g*(Pr.rho[k]-Pr.rho[k-1])*H_tot[i,j,k])

            # diagnostic_variables(Pr.Rd, Pr.Rv, &PV.P.values[0,0,0], &PV.T.values[0,0,0],
            #                      &PV.QT.values[0,0,0],   &self.QL.values[0,0,0], &PV.U.values[0,0,0],
            #                      &PV.V.values[0,0,0],  &self.Divergence.values[0,0,0],&self.KE.values[0,0,0],
            #                      &self.Wp.values[0,0,0], &self.gZ.values[0,0,0], k, nx, ny, nl)
        return
