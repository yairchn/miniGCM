import matplotlib.pyplot as plt
import numpy as np
from math import *
import PrognosticVariables
import DiagnosticVariables

class MicrophysicsNone():
    def __init__(self):
        return
    def initialize(self, Gr, namelist):
        self.QL    = np.zeros((Gr.nx,Gr.ny,Gr.nz), dtype=np.double, order ='c')
        self.QV    = np.zeros((Gr.nx,Gr.ny,Gr.nz), dtype=np.double, order ='c')
        self.QR    = np.zeros((Gr.nx,Gr.ny,Gr.nz), dtype=np.double, order ='c')
        self.dQTdt = np.zeros((Gr.nx,Gr.ny,Gr.nz), dtype=np.double, order ='c')
        self.dTdt  = np.zeros((Gr.nx,Gr.ny,Gr.nz), dtype=np.double, order ='c')
        return
    def update(self, Gr, PV, TS):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Stats):
        return

class Microphysics:
    def __init__(self):
        return

    def initialize(self, namelist, Gr):
        self.PV = PrognosticVariables.PrognosticVariables(Gr)
        self.DV = DiagnosticVariables.DiagnosticVariables(Gr)
        self.max_ss    =  namelist['microphysics']['max_supersaturation']
        self.eps_v     =  namelist['microphysics']['molar_mass_ratio']
        self.MagFormA  =  namelist['microphysics']['Magnus_formula_A']
        self.MagFormB  =  namelist['microphysics']['Magnus_formula_B']
        self.MagFormC  =  namelist['microphysics']['Magnus_formula_C']
        self.QL        = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.QV        = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.QR        = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.dQTdt     = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.dTdt      = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        return

    def initialize_io(self, Stats):
        Stats.add_global_mean('global_mean_QR')
        Stats.add_zonal_mean('zonal_mean_QR')
        Stats.add_meridional_mean('meridional_mean_QR')
        return

    def update(self, Gr, PV, TS):
        for k in range(Gr.n_layers):
            T_cel = PV.T.values[:,:,k] - 273.15
            pv_star = self.MagFormA*np.exp(self.MagFormB*T_cel/(T_cel+self.MagFormC))*100.0
            qv_star = self.eps_v * (1.0 - PV.QT.values[:,:,k]) * pv_star / (PV.P.values[:,:,k] - pv_star)
            self.QL[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            self.QV[:,:,k] = np.minimum(qv_star,PV.QT.values[:,:,k])
            self.QR[:,:,k] = np.clip(self.QL[:,:,k] - self.max_ss*self.QV[:,:,k],0.0, None)
            self.dQTdt[:,:,k] = -self.QR[:,:,k]/TS.dt
            self.dTdt[:,:,k] = -(Gr.Lv/Gr.cp)*self.dQTdt[:,:,k]
        return

    def stats_io(self, TS, Stats):
        Stats.write_global_mean('global_mean_QR', self.QR, TS.t)
        Stats.write_zonal_mean('zonal_mean_QR',self.QR, TS.t)
        Stats.write_meridional_mean('meridional_mean_QR',self.QR, TS.t)
        return

    def io(self, Gr, TS, Stats):
        Stats.write_3D_variable(Gr, int(TS.t), Gr.n_layers, 'Rain',self.QR[:,:,0:Gr.n_layers])
        return
