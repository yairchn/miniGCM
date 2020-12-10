import numpy as np
import scipy as sc
from math import *

class SurfaceBase:
    def __init__(self):
        return
    def initialize(self, Pr, Gr, PV):
        return
    def update(self, Pr, Gr, TS, PV):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class SurfaceNone(SurfaceBase):
    def __init__(self):
        return
    def initialize(self, Pr, Gr, PV):
        return
    def update(self, Pr, Gr, TS, PV):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class Surface_BulkFormula(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    def initialize(self, Pr, Gr, PV):
        self.U_flux  =  np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.V_flux  =  np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.T_flux  =  np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.QT_flux = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.T_surf  = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.QT_surf = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.U_abs = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        return
    def update(self, Pr, Gr, TS, PV, DV):
        U2 = np.multiply(DV.U.values[:,:,Pr.n_layers-1],DV.U.values[:,:,Pr.n_layers-1])
        V2 = np.multiply(DV.V.values[:,:,Pr.n_layers-1],DV.V.values[:,:,Pr.n_layers-1])
        self.U_abs = np.add(U2,V2)
        self.QT_surf = (Pr.qv_star0* Pr.eps_v / PV.P.values[:,:,Pr.n_layers]
                * np.exp(-(Pr.Lv/Pr.Rv)*(1/self.T_surf - 1/Pr.T_0)))

        self.U_flux  = -Pr.Cd*self.U_abs*DV.U.values[:,:,Pr.n_layers-1]
        self.V_flux  = -Pr.Cd*self.U_abs*DV.V.values[:,:,Pr.n_layers-1]
        self.T_flux  = -Pr.Ch*self.U_abs*(PV.T.values[:,:,Pr.n_layers-1]  - self.T_surf)
        self.QT_flux = -Pr.Cq*self.U_abs*(PV.QT.values[:,:,Pr.n_layers-1] - self.QT_surf)
        return
    def initialize_io(self, Stats):
        Stats.add_surface_zonal_mean('zonal_mean_U_flux')
        Stats.add_surface_zonal_mean('zonal_mean_V_flux')
        Stats.add_surface_zonal_mean('zonal_mean_T_flux')
        Stats.add_surface_zonal_mean('zonal_mean_QT_flux')
        return
    def io(self, Pr, TS, Stats):
        Stats.write_surface_zonal_mean('zonal_mean_U_flux',self.U_flux, TS.t)
        Stats.write_surface_zonal_mean('zonal_mean_V_flux',self.V_flux, TS.t)
        Stats.write_surface_zonal_mean('zonal_mean_T_flux',self.T_flux, TS.t)
        Stats.write_surface_zonal_mean('zonal_mean_QT_flux',self.QT_flux, TS.t)
        return
