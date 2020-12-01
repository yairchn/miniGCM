import matplotlib.pyplot as plt
import numpy as np
from math import *

class Microphysics:
    def __init__(self, nu, nz, loc, kind, name, units):
        return

    def initialize(self, Gr):
        self.eps_v     =  namelist['microphysics']['molar_mass_ratio']
        self.MagFormA  =  namelist['microphysics']['Magnus_formula_A']
        self.MagFormB  =  namelist['microphysics']['Magnus_formula_B']
        self.MagFormC  =  namelist['microphysics']['Magnus_formula_C']
        self.QL[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.QV[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.QR[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.dQTdt[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.dTdt[:,:,k]  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        return

    def initialize_io(self, Stats):
        Stats.add_profile('Rain')
        return

    def update(self, Gr, PV, TS):
        for k in range(Gr.n_layers):
            pv_star = self.MagFormA*np.exp((self.MagFormB*(PV.T.values[:,:,k] - 273.15))/((PV.T.values[:,:,k] - 273.15)+self.MagFormC))*100.0
            qv_star = self.eps_v * (1.0 - PV.QT.values[:,:,k]) * self.pv_star[:,:,k] / (PV.P.values[:,:,k] - self.pv_star[:,:,k])
            self.QL[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            # qv = min(qt,qv_star) - need to think about this one
            self.QV[:,:,k] = np.maximum(qv_star,PV.QT.values[:,:,k])
            self.QR[:,:,k] = np.clip(self.QL[:,:,k] - self.max_supersaturation*self.QV[:,:,k],0.0, None)
            self.dQTdt[:,:,k] = -self.QR[:,:,k]/TS.dt
            self.dTdt[:,:,k] = -(Gr.Lv/Gr.cp)*self.dQTdt[:,:,k]
        return

    def io(self, Stats, Ref):
        Stats.write_variable('Rain', self.Rain.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        return
