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
    void diagnostic_variables(double Rd, double Rv, double Omega, double a,
           double* lat,  double* p,  double* T,  double* qt, double* ql, double* u,  double* v,
           double* div, double* ke, double* wp, double* gz, double* uv, double* TT, double* vT,
           double* M, Py_ssize_t k, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

cdef class DiagnosticVariable:
    def __init__(self, nx,ny,nl,n_spec, kind, name, units):
        self.kind = kind
        self.name = name
        self.units = units
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        return

cdef class DiagnosticVariables:
    def __init__(self, Parameters Pr, Grid Gr):
        self.V  = DiagnosticVariable(Pr.nlats, Pr.nlons, 1,   Gr.SphericalGrid.nlm, 'meridional_velocity', 'v','m/s' )
        self.SF  = DiagnosticVariable(Pr.nlats, Pr.nlons, Pr.n_layers,   Gr.SphericalGrid.nlm, 'StreamFunction', 'psi','m^2/s' )
        VₐT, # Equation A9a in the paper - in the code ?
        [U'ₘ, V'ₘ]
        [U'_T, V'_T]
        [q̄ₘ,  q̄ₜ]
        [ψ̄ₘ ?, ψ̄ₜ]
        [ψₘ', ψₜ']
        return

    cpdef initialize_io(self, Parameters Pr, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_KE')
        Stats.add_zonal_mean('zonal_mean_U')
        Stats.add_zonal_mean('zonal_mean_V')
        Stats.add_meridional_mean('meridional_mean_U')
        return

    cpdef stats_io(self, Parameters Pr, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_KE', self.KE.values)
        Stats.write_zonal_mean('zonal_mean_U',self.U.values)
        Stats.write_meridional_mean('meridional_mean_U',self.U.values)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'meridional_velocity', self.V.values)
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'mean PV',             self.Q.values)
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

        # assuming f*psi=geopotential for the heat flux term
        RHS_v_t = ((1/Pr.Omega)*((1/Pr.Earth_rad^2)*Pr.cos_lat_sqr.* d2_heatflxmiu_dmiu2  +  (Gr.miu/Pr.Earth_rad) *d_momflxT_dmiu -
                   (D.r_mean/2)*(D.miu).*(UM - UT) + D.alpha_mean*(D.miu).*(D.Ut_rad - UT) ) - np.transpose(Gr.DH_term.))

        # the inverse of the operator acting on vt
        inv_op_vt = inv_operator_vt(Pr.miu, Pr.Omega, Pr.epsilon, zetaM_mean, Pr.Der2_dir, Pr.cos_lat_sqr)
        # the baroclinic component of the meridional velocity
        Vt = inv_op_vt * (np.transpose(RHS_v_t.))




        self.Wp.values.base[:,:,0] = np.zeros_like(self.Wp.values[:,:,0])
        self.gZ.values.base[:,:,nl] = np.zeros_like(self.Wp.values[:,:,0])
        for k in range(nl):
            self.U.values.base[:,:,k], self.V.values.base[:,:,k] = Gr.SphericalGrid.getuv(
                         PV.Vorticity.spectral.base[:,k],PV.Divergence.spectral.base[:,k])
            with nogil:
                diagnostic_variables(Pr.Rd, Pr.Rv, Pr.Omega, Pr.rsphere, &Gr.lat[0,0], &PV.P.values[0,0,0], &PV.T.values[0,0,0],
                                     &PV.QT.values[0,0,0],   &self.QL.values[0,0,0], &self.U.values[0,0,0],
                                     &self.V.values[0,0,0],  &PV.Divergence.values[0,0,0],
                                     &self.KE.values[0,0,0], &self.Wp.values[0,0,0], &self.gZ.values[0,0,0],
                                     &self.UV.values[0,0,0], &self.TT.values[0,0,0], &self.VT.values[0,0,0],
                                     &self.M.values[0,0,0], k, nx, ny, nl)
        return
