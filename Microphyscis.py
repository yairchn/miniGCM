import matplotlib.pyplot as plt
import numpy as np
from math import *

class Microphysics:
    def __init__(self, nu, nz, loc, kind, name, units):
        return

    def initialize(self, Gr):
        self.eps_v =  namelist['thermodynamics']['molar_mass_ratio']
        self.QL[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.QV[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.QR[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.dQTdt[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        self.dTdt[:,:,k] = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),dtype=np.double, order='c')
        return

    def initialize_io(self, Stats):
        Stats.add_profile('Rain')
        return

    def update(self, Gr, PV, TS):
        for k in range(Gr.n_layers):
            pv_star = 6.1094*np.exp((17.625*(PV.T.values[:,:,k] - 273.15))/((PV.T.values[:,:,k] - 273.15)+243.04))*100.0
            qv_star = self.eps_v * (1.0 - PV.QT.values[:,:,k]) * self.pv_star[:,:,k] / (PV.P.values[:,:,k] - self.pv_star[:,:,k])
            self.QL[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
            # qv = min(qt,qv_star) - need to think about this one
            # self.QV[:,:,k] = np.clip(qv_star - PV.QT.values[:,:,k],0.0, None)
            self.QR[:,:,k] = np.clip(self.QL[:,:,k] - self.max_supersaturation*self.QV[:,:,k],0.0, None)
            self.dQTdt[:,:,k] = -self.QR[:,:,k]/TS.dt
            self.dTdt[:,:,k] = -(Gr.Lv/Gr.cp)*self.dQTdt[:,:,k]
        return

    def io(self, Stats, Ref):
        Stats.write_variable('Rain', self.Rain.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        return
