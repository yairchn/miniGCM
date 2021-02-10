import cython
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np
from math import *
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport pow, fmax, exp
import pylab as plt

cdef extern from "microphysics_functions.h":
    void microphysics_cutoff(double cp, double dt, double Rv, double Lv, double T_0, double rho_w,
           double g, double max_ss, double qv_star0, double eps_v, double* p, double* T,
           double* qt, double* ql, double* T_mp, double* qt_mp, double* rain_rate,
           Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

cdef class MicrophysicsBase:
    def __init__(self):
        return
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class MicrophysicsNone(MicrophysicsBase):
    def __init__(self):
        MicrophysicsBase.__init__(self)
        return
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        PV.QT.mp_tendency = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.T.mp_tendency  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        return
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class MicrophysicsCutoff(MicrophysicsBase):
    def __init__(self):
        MicrophysicsBase.__init__(self)
        return

    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        cdef:
            double [:,:] P_half
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        Pr.max_ss    =  namelist['microphysics']['max_supersaturation']
        Pr.rho_w     =  namelist['microphysics']['water_density']
        for k in range(Pr.n_layers):
            P_half = np.multiply(np.add(PV.P.values[:,:,k],PV.P.values[:,:,k+1]),0.5)
            # Clausius–Clapeyron equation based saturation
            qv_star = np.multiply(np.divide(np.multiply(Pr.qv_star0,Pr.eps_v),P_half),
                np.exp(-np.multiply(np.divide(Pr.Lv,Pr.Rv),np.subtract(np.divide(1.0,PV.T.values[:,:,k]),np.divide(1.0,Pr.T_0)))))

            DV.QL.values.base[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            if np.max(DV.QL.values.base[:,:,k])>0.0:
                print('============| WARNING |===============')
                print('ql is non zero in the initial state of layer number', k+1)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_dQTdt')
        Stats.add_zonal_mean('zonal_mean_dQTdt')
        Stats.add_meridional_mean('meridional_mean_dQTdt')
        Stats.add_surface_zonal_mean('zonal_mean_RainRate')
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            double P_half, qv_star, denom

        with nogil:
            microphysics_cutoff(Pr.cp, TS.dt, Pr.Rv, Pr.Lv, Pr.T_0, Pr.rho_w,
                                Pr.g, Pr.max_ss, Pr.qv_star0, Pr.eps_v, &PV.P.values[0,0,0],
                                &PV.T.values[0,0,0], &PV.QT.values[0,0,0], &DV.QL.values[0,0,0],
                                &PV.T.mp_tendency[0,0,0], &PV.QT.mp_tendency[0,0,0], &self.RainRate[0,0],
                                nx, ny, nl)

        return

    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_dQTdt', PV.QT.mp_tendency)
        Stats.write_zonal_mean('zonal_mean_dQTdt',PV.QT.mp_tendency)
        Stats.write_meridional_mean('meridional_mean_dQTdt',PV.QT.mp_tendency)
        Stats.write_surface_zonal_mean('zonal_mean_RainRate',np.mean(self.RainRate,axis=1))
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_2D_variable(Pr, int(TS.t)             , 'Rain_Rate',self.RainRate)
        return
