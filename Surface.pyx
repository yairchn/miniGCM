import cython
from Grid cimport Grid
from DiagnosticVariables cimport DiagnosticVariables
from math import *
import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
import sphericalForcing as spf
import time
from TimeStepping cimport TimeStepping
import sys
from Parameters cimport Parameters
from libc.math cimport exp, pow, sqrt

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
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
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
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
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
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            double U_abs, z_a

        with nogil:
            for i in range(nx):
                for j in range(ny):
                    U_abs = sqrt(pow(DV.U.values[i,j,nl-1],2.0) + pow(DV.V.values[i,j,nl-1],2.0))
                    self.QT_surf[i,j] = Pr.qv_star0*Pr.eps_v/PV.P.values[i,j,nl]*exp(-Pr.Lv/Pr.Rv*(1.0/self.T_surf[i,j] - 1.0/Pr.T_0))
                    z_a = DV.gZ.values[i,j,nl-1]/Pr.g
                    DV.U.SurfaceFlux[i,j]  = -Pr.Cd/z_a*U_abs*DV.U.values[i,j,nl-1]
                    DV.V.SurfaceFlux[i,j]  = -Pr.Cd/z_a*U_abs*DV.V.values[i,j,nl-1]
                    PV.T.SurfaceFlux[i,j]  = -Pr.Ch/z_a*U_abs*(PV.T.values[i,j,nl-1] - self.T_surf[i,j])
                    PV.QT.SurfaceFlux[i,j] = -Pr.Cq/z_a*U_abs*(PV.QT.values[i,j,nl-1] - self.QT_surf[i,j])

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_surface_zonal_mean('zonal_mean_T_surf')
        Stats.add_surface_zonal_mean('zonal_mean_QT_surf')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_surface_zonal_mean('zonal_mean_QT_surf', np.mean(self.T_surf, axis=1))
        Stats.write_surface_zonal_mean('zonal_mean_QT_surf', np.mean(self.QT_surf, axis=1))
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_2D_variable(Pr, TS.t,  'T_surf', self.T_surf)
        Stats.write_2D_variable(Pr, TS.t,  'QT_surf', self.QT_surf)
        return
