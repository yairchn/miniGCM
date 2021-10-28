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
    void vertical_turbulent_flux(double cp, double lv, double g, double Ch, double Cq, double kappa, double p_ref,
                                 double Ppbl, double Pstrato, double* p, double* gz,
                                 double* T, double* qt, double* u, double* v, double* wE,
                                 double* wqt, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

def TurbulenceFactory(namelist):
    if namelist['turbulence']['turbulence_model'] == 'None':
        return TurbulenceNone(namelist)
    elif namelist['turbulence']['turbulence_model'] == 'DownGradient':
        return DownGradientTurbulence(namelist)
    else:
        print('case not recognized')
    return

cdef class TurbulenceBase:
    def __init__(self, namelist):
        return
    cpdef initialize(self, Parameters Pr, namelist):
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
    def __init__(self, namelist):
        TurbulenceBase.__init__(self, namelist)
        return
    cpdef initialize(self, Parameters Pr, namelist):
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
    def __init__(self, namelist):
        TurbulenceBase.__init__(self, namelist)
        return
    cpdef initialize(self, Parameters Pr, namelist):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers

        Pr.Pstrato = namelist['turbulence']['stratospheric_pressure']
        Pr.Ppbl = namelist['turbulence']['boundary_layer_top_pressure']
        Pr.Dh = namelist['turbulence']['sensible_heat_transfer_coeff']
        Pr.Dq = namelist['turbulence']['latent_heat_transfer_coeff']
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
            vertical_turbulent_flux(Pr.cp, Pr.Lv, Pr.g, Pr.Dh, Pr.Dq, Pr.kappa, Pr.p_ref, Pr.Ppbl, Pr.Pstrato,
                              &PV.P.values[0,0,0],&DV.gZ.values[0,0,0],
                              &DV.T.values[0,0,0],&PV.QT.values[0,0,0],
                              &DV.U.values[0,0,0],&DV.V.values[0,0,0],
                              &PV.E.TurbFlux[0,0,0],&PV.QT.TurbFlux[0,0,0],
                               nx, ny, nl)
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        # Stats.add_zonal_mean('zonal_mean_QT_Turb')
        # Stats.add_zonal_mean('zonal_mean_T_Turb')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        # Stats.write_zonal_mean('zonal_mean_QT_Turb', PV.QT.TurbFlux)
        # Stats.write_zonal_mean('zonal_mean_T_Turb', PV.T.TurbFlux)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        # Stats.write_3D_variable(Pr, TS.t,  'QT_Turb',  PV.QT.TurbFlux)
        # Stats.write_3D_variable(Pr, TS.t,  'T_Turb',PV.T.TurbFlux)
        return
