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

cdef extern from "surface_functions.h":
    void surface_bulk_formula(double g, double Rv, double Lv, double T_0, double Ch, double Cq,
                              double Cd, double qv_star0, double eps_v, double* p, double* gz, double* T,
                              double* qt, double* T_surf, double* u, double* v, double* u_surf_flux,
                              double* v_surf_flux, double* T_surf_flux, double* qt_surf_flux,
                              Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil


cdef class SurfaceBase:
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
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class SurfaceNone(SurfaceBase):
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
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class SurfaceBulkFormula(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        Pr.Cd = namelist['surface']['momentum_transfer_coeff']
        Pr.Ch = namelist['surface']['sensible_heat_transfer_coeff']
        Pr.Cq = namelist['surface']['latent_heat_transfer_coeff']
        Pr.dT_s = namelist['surface']['surface_temp_diff']
        Pr.T_min = namelist['surface']['surface_temp_min']
        Pr.dphi_s = namelist['surface']['surface_temp_lat_dif']
        self.T_surf  = np.multiply(Pr.dT_s,np.exp(-0.5*np.power(Gr.lat,2.0)/Pr.dphi_s**2.0)) + Pr.T_min
        self.QT_surf = np.multiply(np.divide(Pr.qv_star0*Pr.eps_v, PV.P.values[:,:,Pr.n_layers]),
                       np.exp(-np.multiply(Pr.Lv/Pr.Rv,np.subtract(np.divide(1,self.T_surf) , np.divide(1,Pr.T_0) ))))
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nx
            Py_ssize_t ny = Pr.ny
            Py_ssize_t nl = Pr.n_layers

        with nogil:
            surface_bulk_formula(Pr.g, Pr.Rv, Pr.Lv, Pr.T_0, Pr.Ch, Pr.Cq,
                              Pr.Cd, Pr.qv_star0, Pr.eps_v, &PV.P.values[0,0,0],
                              &DV.gZ.values[0,0,0], &PV.T.values[0,0,0],
                              &PV.QT.values[0,0,0], &self.T_surf[0,0],
                              &PV.U.values[0,0,0], &PV.V.values[0,0,0],
                              &PV.U.SurfaceFlux[0,0], &PV.V.SurfaceFlux[0,0],
                              &PV.T.SurfaceFlux[0,0], &PV.QT.SurfaceFlux[0,0],
                              nx, ny, nl)

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_surface_axisymmetric_mean('axisymmetric_mean_T_surf')
        Stats.add_surface_axisymmetric_mean('axisymmetric_mean_QT_surf')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_surface_axisymmetric_mean('axisymmetric_mean_T_surf', np.mean(self.T_surf, axis=1))
        Stats.write_surface_axisymmetric_mean('axisymmetric_mean_QT_surf', np.mean(self.QT_surf, axis=1))
        return

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_2D_variable(Pr, Gr, TS.t,  'T_surf', self.T_surf)
        Stats.write_2D_variable(Pr, Gr, TS.t,  'QT_surf', self.QT_surf)
        return
