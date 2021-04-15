import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
from DiagnosticVariables cimport DiagnosticVariables
import numpy as np
cimport numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

cdef extern from "turbulence_functions.h":
    void vertical_turbulent_flux(double g, double c_e, double kappa, double p_ref,
                                 double Ppbl, double Pstrato, double* p, double* gz,
                                 double* T, double* qt, double* u, double* v, double* wTh,
                                 double* wqt, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil





cdef class TurbulenceBase:
    def __init__(self):
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class TurbulenceNone(TurbulenceBase):
    def __init__(self):
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class DownGradientTurbulence(TurbulenceBase):
    def __init__(self):
        TurbulenceBase.__init__(self)
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers

        Pr.Ce = namelist['turbulence']['']
        Pr.Pstrato = namelist['turbulence']['']
        Pr.Ppbl = namelist['turbulence']['']
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers

        with nogil:
            vertical_turbulent_flux(Pr.g, Pr.Ce, Pr.kappa, Pr.p_ref, Pr.Ppbl, Pr.Pstrato,
                              &PV.P.values[0,0,0],&DV.gZ.values[0,0,0],
                              &PV.T.values[0,0,0],&PV.QT.values[0,0,0],
                              &DV.U.values[0,0,0],&DV.V.values[0,0,0],
                              &PV.T.TurbFlux[0,0,0],&PV.QT.TurbFlux[0,0,0],
                               nx, ny, nl)

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        # Stats.add_surface_zonal_mean('zonal_mean_T_surf')
        # Stats.add_surface_zonal_mean('zonal_mean_QT_surf')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        # Stats.write_surface_zonal_mean('zonal_mean_QT_surf', self.T_surf)
        # Stats.write_surface_zonal_mean('zonal_mean_QT_surf', self.QT_surf)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        # Stats.write_2D_variable(Pr, TS.t,  'T_surf', self.T_surf)
        # Stats.write_2D_variable(Pr, TS.t,  'QT_surf', self.QT_surf)
        return
