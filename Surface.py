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

class SurfaceBulkFormula(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    def initialize(self, Pr, Gr, PV, namelist):
        self.U_flux  =  np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.V_flux  =  np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.T_flux  =  np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.QT_flux = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.T_surf  = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.U_abs   = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        Pr.Cd = namelist['surface']['momentum_transfer_coeff']
        Pr.Ch = namelist['surface']['sensible_heat_transfer_coeff']
        Pr.Cq = namelist['surface']['latent_heat_transfer_coeff']
        Pr.dT_s = namelist['surface']['surface_temp_diff']
        Pr.T_min = namelist['surface']['surface_temp_min']
        Pr.dphi_s = namelist['surface']['surface_temp_lat_dif']
        self.T_surf = Pr.dT_s*np.exp(-np.power(-0.5*Gr.lat,2.0)/Pr.dphi_s**2.0) + Pr.T_min
        self.QT_surf = (Pr.qv_star0* Pr.eps_v / PV.P.values[:,:,Pr.n_layers]
                * np.exp(-(Pr.Lv/Pr.Rv)*(1/self.T_surf - 1/Pr.T_0)))
        return
    def update(self, Pr, Gr, TS, PV, DV):
        U2 = np.multiply(DV.U.values[:,:,Pr.n_layers-1],DV.U.values[:,:,Pr.n_layers-1])
        V2 = np.multiply(DV.V.values[:,:,Pr.n_layers-1],DV.V.values[:,:,Pr.n_layers-1])
        self.U_abs = np.add(U2,V2)

        self.U_flux  = -Pr.Cd*self.U_abs*DV.U.values[:,:,Pr.n_layers-1]
        self.V_flux  = -Pr.Cd*self.U_abs*DV.V.values[:,:,Pr.n_layers-1]
        self.T_flux  = -Pr.Ch*self.U_abs*(PV.T.values[:,:,Pr.n_layers-1]  - self.T_surf)
        self.QT_flux = -Pr.Cq*self.U_abs*(PV.QT.values[:,:,Pr.n_layers-1] - self.QT_surf)
        print('U_flux',np.max(np.abs(self.U_flux)))
        print('V_flux',np.max(np.abs(self.V_flux)))
        print('T_flux',np.max(np.abs(self.T_flux)),np.max( self.T_surf))
        print('QT_flux',np.max(np.abs(self.QT_flux)), np.max(self.QT_surf))

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
