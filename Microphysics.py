import matplotlib.pyplot as plt
import numpy as np
from math import *
import PrognosticVariables
import DiagnosticVariables

class MicrophysicsBase:
    def __init__(self):
        return
    def initialize(self, Pr, namelist):
        return
    def update(self, Pr, Gr, PV, TS):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class MicrophysicsNone(MicrophysicsBase):
    def __init__(self):
        MicrophysicsBase.__init__(self)
        return
    def initialize(self, Pr, namelist):
        self.dQTdt = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.dTdt  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        return
    def update(self, Pr, Gr, PV, TS):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class MicrophysicsCutoff(MicrophysicsBase):
    def __init__(self):
        MicrophysicsBase.__init__(self)
        return

    def initialize(self, Pr, namelist):
        self.QL        = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.QV        = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.QR        = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.dQTdt     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.dTdt      = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        Pr.max_ss    =  namelist['microphysics']['max_supersaturation']
        Pr.eps_v     =  namelist['microphysics']['molar_mass_ratio']
        Pr.MagFormA  =  namelist['microphysics']['Magnus_formula_A']
        Pr.MagFormB  =  namelist['microphysics']['Magnus_formula_B']
        Pr.MagFormC  =  namelist['microphysics']['Magnus_formula_C']
        Pr.rho_w     =  namelist['microphysics']['water_density']
        return

    def initialize_io(self, Stats):
        Stats.add_global_mean('global_mean_QR')
        Stats.add_zonal_mean('zonal_mean_QR')
        Stats.add_meridional_mean('meridional_mean_QR')
        return

    def update(self, Pr, Gr, TS, PV):
        for k in range(Pr.n_layers):
            # qv_star = (Pr.eps_v / (PV.P.values[:,:,k] + PV.P.values[:,:,k+1])
            #     * np.exp(-(Pr.Lv/Pr.Rv)*(1/PV.T.values[:,:,k] - 1/Pr.T_0)))
            # self.QL[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            # self.QV[:,:,k] = np.minimum(qv_star,PV.QT.values[:,:,k])
            # self.QR[:,:,k] = PV.QT.values[:,:,k] - (1.0+Pr.max_ss)*qv_star
            # self.QR[:,:,k] = np.clip(self.QR[:,:,k],0.0, None)
            # print('exp arg')
            # print(-(Pr.Lv/Pr.Rv)*(1/PV.T.values[:,:,k] - 1/Pr.T_0))
            # print('exp val')
            # print(np.exp(-(Pr.Lv/Pr.Rv)*(1/PV.T.values[:,:,k] - 1/Pr.T_0)))
            # print(k, np.max(self.QR[:,:,k]), np.max(PV.QT.values[:,:,k]), np.max(qv_star))
            # # self.QR[:,:,k] = np.clip(self.QL[:,:,k] - Pr.max_ss*self.QV[:,:,k],0.0, None)
            # self.dQTdt[:,:,k] = -self.QR[:,:,k]/TS.dt
            # self.dTdt[:,:,k]  = -(Pr.Lv/Pr.cp)*self.dQTdt[:,:,k]

            # magnus formula alternative
            T_cel = PV.T.values[:,:,k] - 273.15
            pv_star = Pr.MagFormA*np.exp(Pr.MagFormB*T_cel/(T_cel+Pr.MagFormC))*100.0
            qv_star = Pr.eps_v * (1.0 - PV.QT.values[:,:,k]) * pv_star / (PV.P.values[:,:,k] - pv_star)
            self.QL[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            self.QV[:,:,k] = np.minimum(qv_star,PV.QT.values[:,:,k])
            self.QR[:,:,k] = PV.QT.values[:,:,k] - (1.0+Pr.max_ss)*qv_star
            self.QR[:,:,k] = np.clip(self.QR[:,:,k],0.0, None)
            self.dQTdt[:,:,k] = -self.QR[:,:,k]/TS.dt
            self.dTdt[:,:,k]  = -(Pr.Lv/Pr.cp)*self.dQTdt[:,:,k]
        self.RainRate = np.divide(np.sum(self.dQTdt,2),
                 Pr.rho_w*Pr.g*(PV.P.values[:,:,Pr.n_layers]-PV.P.values[:,:,0]))
        return

    def stats_io(self, TS, Stats):
        Stats.write_global_mean('global_mean_QR', self.QR, TS.t)
        Stats.write_zonal_mean('zonal_mean_QR',self.QR, TS.t)
        Stats.write_meridional_mean('meridional_mean_QR',self.QR, TS.t)
        return

    def io(self, Pr, TS, Stats):
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Rain',self.QR[:,:,0:Pr.n_layers])
        Stats.write_2D_variable(Pr, int(TS.t)             , 'Rain Rate',self.RainRate)
        return
