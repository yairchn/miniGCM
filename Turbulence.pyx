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
    void down_gradient_turbulent_flux(double g, double Rv, double Lv, double T_0, double Ch, double Cq,
                              double Cd, double qv_star0, double eps_v, double* p, double* gz, double* T,
                              double* qt, double* T_surf, double* u, double* v, double* u_surf_flux,
                              double* v_surf_flux, double* T_surf_flux, double* qt_surf_flux,
                              Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil


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
        SurfaceBase.__init__(self)
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers

        Pr.Ce = namelist['turbulence']['']
        Pr.pstrato = namelist['turbulence']['']
        Pr.ppbl = namelist['turbulence']['']
        # eq. (17) Tatcher and Jablonowski 2016
        self.K_e = np.zeros(?)
        for i in range(nx):
            for j in range(ny):
                for k in range(nl):
                    self.K_e[i,j,k] = Pr.Ce*
                    self.QT_surf = np.multiply(np.divide(Pr.qv_star0*Pr.eps_v, PV.P.values[:,:,Pr.n_layers]),
                                   np.exp(-np.multiply(Pr.Lv/Pr.Rv,np.subtract(np.divide(1,self.T_surf) , np.divide(1,Pr.T_0) ))))
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
            down_gradient_turbulent_flux(Pr.g, Pr.Rv, Pr.Lv, Pr.T_0, Pr.Ch, Pr.Cq,
                              Pr.Cd, Pr.qv_star0, Pr.eps_v, &PV.P.values[0,0,0],
                              &DV.gZ.values[0,0,0], &PV.T.values[0,0,0],
                              &PV.QT.values[0,0,0], &self.T_surf[0,0],
                              &DV.U.values[0,0,0], &DV.V.values[0,0,0],
                              &DV.U.SurfaceFlux[0,0], &DV.V.SurfaceFlux[0,0],
                              &PV.T.SurfaceFlux[0,0], &PV.QT.SurfaceFlux[0,0],
                              nx, ny, nl)

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_surface_zonal_mean('zonal_mean_T_surf')
        Stats.add_surface_zonal_mean('zonal_mean_QT_surf')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_surface_zonal_mean('zonal_mean_QT_surf', self.T_surf)
        Stats.write_surface_zonal_mean('zonal_mean_QT_surf', self.QT_surf)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_2D_variable(Pr, TS.t,  'T_surf', self.T_surf)
        Stats.write_2D_variable(Pr, TS.t,  'QT_surf', self.QT_surf)
        return
