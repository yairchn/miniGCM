import cython
import matplotlib.pyplot as plt
import numpy as np
from math import *
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport pow, fmax, exp

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
        # PV.QT.mp_tendency = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        # PV.T.mp_tendency  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
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
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            double P_half, qv_star, denom

        with nogil:
            for i in range(nx):
                for j in range(ny):
                    self.RainRate[i,j] = 0.0
                    for k in range(nl):
                        P_half = 0.5*(PV.P.values[i,j,k]+PV.P.values[i,j,k+1])
                        qv_star = (Pr.qv_star0*Pr.eps_v/P_half)*exp(-Pr.Lv/Pr.Rv*(1.0/PV.T.values[i,j,k]-1.0/Pr.T_0))
                        denom = (1.0+Pr.Lv**2.0/Pr.cp/Pr.Rv*qv_star/pow(PV.T.values[i,j,k],2.0))*TS.dt
                        DV.QL.values[i,j,k] = fmax(PV.QT.values[i,j,k] - qv_star,0.0)
                        PV.T.mp_tendency[i,j,k]  =  Pr.Lv/Pr.cp*DV.QL.values[i,j,k]/denom
                        PV.QT.mp_tendency[i,j,k] = -fmax(PV.QT.values[i,j,k] - (1.0+Pr.max_ss)*qv_star, 0.0)/denom
                        self.RainRate[i,j] -= (PV.QT.mp_tendency[i,j,k]/Pr.rho_w*Pr.g*(PV.P.values[i,j,nl]-PV.P.values[i,j,0]))/(nl+1)

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
