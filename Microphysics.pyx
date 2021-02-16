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

# def MicrophysicsFactory(namelist):
#     if namelist['meta']['casename'] == 'HeldSuarez':
#         return MicrophysicsNone()
#     elif namelist['meta']['casename'] == 'HeldSuarezMoist':
#         return MicrophysicsCutoff()
#     else:
#         print('Microphysics sceme not recognized')
#     return

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
        PV.QT.mp_tendency = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        PV.T.mp_tendency  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
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
        # self.QL        = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        Pr.max_ss    =  namelist['microphysics']['max_supersaturation']
        # Pr.MagFormA  =  namelist['microphysics']['Magnus_formula_A']
        # Pr.MagFormB  =  namelist['microphysics']['Magnus_formula_B']
        # Pr.MagFormC  =  namelist['microphysics']['Magnus_formula_C']
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
            Py_ssize_t k
            Py_ssize_t nl = Pr.n_layers

        for k in range(nl):

            # magnus formula alternative
            # T_cel = PV.T.values[:,:,k] - 273.15
            # pv_star = Pr.MagFormA*np.exp(Pr.MagFormB*T_cel/(T_cel+Pr.MagFormC))*100.0
            # qv_star = Pr.eps_v * (1.0 - PV.QT.values[:,:,k]) * pv_star / (PV.P.values[:,:,k] - pv_star)

            # Clausius–Clapeyron equation based saturation
            P_half = np.multiply(np.add(PV.P.values[:,:,k],PV.P.values[:,:,k+1]),0.5)

            qv_star = np.multiply(np.divide(np.multiply(Pr.qv_star0,Pr.eps_v),P_half),
                np.exp(-np.multiply(Pr.Lv/Pr.Rv,np.subtract(np.divide(1.0,PV.T.values[:,:,k]),np.divide(1.0,Pr.T_0)))))

            DV.QL.values.base[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            denom = np.add(1.0,np.multiply((Pr.Lv**2.0/Pr.cp/Pr.Rv),np.divide(qv_star,np.power(PV.T.values[:,:,k],2.0))))
            PV.QT.mp_tendency.base[:,:,k] = -np.clip(np.divide(np.divide(np.subtract(PV.QT.values[:,:,k], np.multiply(1.0+Pr.max_ss,qv_star)),denom),TS.dt), 0.0, None)
            PV.T.mp_tendency.base[:,:,k]  =  np.clip(np.multiply(Pr.Lv/Pr.cp,np.divide(np.divide(np.subtract(PV.QT.values[:,:,k], qv_star),denom),TS.dt)), 0.0, None)

        self.RainRate = -np.divide(np.sum(PV.QT.mp_tendency,2),
                 Pr.rho_w*Pr.g*(np.subtract(PV.P.values[:,:,nl],PV.P.values[:,:,0])))
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
