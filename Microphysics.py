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
    def stats_io(self, TS, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class MicrophysicsCutoff(MicrophysicsBase):
    def __init__(self):
        MicrophysicsBase.__init__(self)
        return

    def initialize(self, Pr, PV, namelist):
        self.QL        = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.QV        = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.QR        = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.dQTdt     = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.dTdt      = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),dtype=np.double, order='c')
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        Pr.max_ss    =  namelist['microphysics']['max_supersaturation']
        Pr.MagFormA  =  namelist['microphysics']['Magnus_formula_A']
        Pr.MagFormB  =  namelist['microphysics']['Magnus_formula_B']
        Pr.MagFormC  =  namelist['microphysics']['Magnus_formula_C']
        Pr.rho_w     =  namelist['microphysics']['water_density']
        for k in range(Pr.n_layers):
            # Clausius–Clapeyron equation based saturation
            qv_star = (Pr.qv_star0* Pr.eps_v / (PV.P.values[:,:,k] + PV.P.values[:,:,k+1])
                * np.exp(-(Pr.Lv/Pr.Rv)*(1/PV.T.values[:,:,k] - 1/Pr.T_0)))

            self.QL[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            if np.max(self.QL[:,:,k])>0.0:
                print('============| WARNING |===============')
                print('ql is non zero in initial state of layer ', k)
        self.RainRate = np.divide(np.sum(-self.dQTdt,2),
                 Pr.rho_w*Pr.g*(PV.P.values[:,:,Pr.n_layers]-PV.P.values[:,:,0]))
        return

    def initialize_io(self, Stats):
        Stats.add_global_mean('global_mean_QL')
        Stats.add_zonal_mean('zonal_mean_QL')
        Stats.add_meridional_mean('meridional_mean_QL')
        Stats.add_global_mean('global_mean_dQTdt')
        Stats.add_zonal_mean('zonal_mean_dQTdt')
        Stats.add_meridional_mean('meridional_mean_dQTdt')
        Stats.add_surface_zonal_mean('zonal_mean_RainRate')
        return

    def update(self, Pr, Gr, TS, PV):
        for k in range(Pr.n_layers):

            # magnus formula alternative
            # T_cel = PV.T.values[:,:,k] - 273.15
            # pv_star = Pr.MagFormA*np.exp(Pr.MagFormB*T_cel/(T_cel+Pr.MagFormC))*100.0
            # qv_star = Pr.eps_v * (1.0 - PV.QT.values[:,:,k]) * pv_star / (PV.P.values[:,:,k] - pv_star)

            # Clausius–Clapeyron equation based saturation
            qv_star = (Pr.qv_star0* Pr.eps_v / (PV.P.values[:,:,k] + PV.P.values[:,:,k+1])
                * np.exp(-(Pr.Lv/Pr.Rv)*(1/PV.T.values[:,:,k] - 1/Pr.T_0)))
            self.QL[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            denom = 1.0+(Pr.Lv**2.0/Pr.cp/Pr.Rv)*np.divide(qv_star,np.power(PV.T.values[:,:,k],2.0))
            self.dQTdt[:,:,k] = -np.clip(((PV.QT.values[:,:,k] - (1.0+Pr.max_ss)*qv_star) /denom)/TS.dt, 0.0, None)
            self.dTdt[:,:,k]  =  np.clip((Pr.Lv/Pr.cp)*((PV.QT.values[:,:,k] -  qv_star)/denom)/TS.dt, 0.0, None)

        self.RainRate = np.divide(np.sum(-self.dQTdt,2),
                 Pr.rho_w*Pr.g*(PV.P.values[:,:,Pr.n_layers]-PV.P.values[:,:,0]))
        return

    def stats_io(self, TS, Stats):
        Stats.write_global_mean('global_mean_QL', self.QL, TS.t)
        Stats.write_zonal_mean('zonal_mean_QL',self.QL, TS.t)
        Stats.write_meridional_mean('meridional_mean_QL',self.QL, TS.t)
        Stats.write_global_mean('global_mean_dQTdt', self.dQTdt, TS.t)
        Stats.write_zonal_mean('zonal_mean_dQTdt',self.dQTdt, TS.t)
        Stats.write_meridional_mean('meridional_mean_dQTdt',self.dQTdt, TS.t)
        Stats.write_surface_zonal_mean('zonal_mean_RainRate',self.RainRate, TS.t)
        return

    def io(self, Pr, TS, Stats):
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'Liquid_Water',self.QL[:,:,0:Pr.n_layers])
        Stats.write_3D_variable(Pr, int(TS.t), Pr.n_layers, 'dQTdt',self.dQTdt[:,:,0:Pr.n_layers])
        Stats.write_2D_variable(Pr, int(TS.t)             , 'Rain_Rate',self.RainRate)
        return
