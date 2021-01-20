import cython
from Grid cimport Grid
from DiagnosticVariables cimport DiagnosticVariables
from math import *
import matplotlib.pyplot as plt
import numpy as np
from NetCDFIO cimport NetCDFIO_Stats
from PrognosticVariables cimport PrognosticVariables
import scipy as sc
import sphericalForcing as spf
import time
from TimeStepping cimport TimeStepping
import sys
from Parameters cimport Parameters

cdef class SurfaceBase:
    def __init__(self):
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class SurfaceNone(SurfaceBase):
    def __init__(self):
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class SurfaceBulkFormula(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
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
        self.T_surf  = np.multiply(Pr.dT_s,np.exp(-0.5*np.power(Gr.lat,2.0)/Pr.dphi_s**2.0)) + Pr.T_min
        self.QT_surf = np.multiply(np.divide(Pr.qv_star0*Pr.eps_v, PV.P.values[:,:,Pr.n_layers]),
                       np.exp(-np.multiply(Pr.Lv/Pr.Rv,np.subtract(np.divide(1,self.T_surf) , np.divide(1,Pr.T_0) ))))
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t nl = Pr.n_layers
            double [:,:] z_a
        U2 = np.multiply(DV.U.values[:,:,nl-1],DV.U.values[:,:,nl-1])
        V2 = np.multiply(DV.V.values[:,:,nl-1],DV.V.values[:,:,nl-1])
        self.U_abs = np.sqrt(np.add(U2,V2))
        self.QT_surf = np.multiply(np.divide(Pr.qv_star0*Pr.eps_v, PV.P.values[:,:,Pr.n_layers]),
                       np.exp(-np.multiply(Pr.Lv/Pr.Rv,np.subtract(np.divide(1,self.T_surf) , np.divide(1,Pr.T_0) ))))
        # T_va = np.multiply(self.T_surf,np.add(1.0,np.multiply(0.608,PV.QT.values[:,:,nl-1])))
        # z_a = np.multiply(np.multiply(Pr.Rd/Pr.g,self.T_surf),np.divide(np.subtract(
        #                   np.log(PV.P.values[:,:,Pr.n_layers])-np.log(PV.P.values[:,:,nl-1])),2.0))
        z_a = np.divide(DV.gZ.values[:,:,nl-1],Pr.g)
        DV.U.SurfaceFlux  = -np.multiply(np.multiply(np.divide(Pr.Cd,z_a),self.U_abs),DV.U.values[:,:,Pr.n_layers-1])
        DV.V.SurfaceFlux  = -np.multiply(np.multiply(np.divide(Pr.Cd,z_a),self.U_abs),DV.V.values[:,:,Pr.n_layers-1])
        PV.T.SurfaceFlux  = -np.multiply(np.multiply(np.divide(Pr.Ch,z_a),self.U_abs),np.subtract(PV.T.values[:,:,Pr.n_layers-1] , self.T_surf))
        PV.QT.SurfaceFlux = -np.multiply(np.multiply(np.divide(Pr.Cq,z_a),self.U_abs),np.subtract(PV.QT.values[:,:,Pr.n_layers-1], self.QT_surf))
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_surface_zonal_mean('zonal_mean_U_flux')
        Stats.add_surface_zonal_mean('zonal_mean_V_flux')
        Stats.add_surface_zonal_mean('zonal_mean_T_flux')
        Stats.add_surface_zonal_mean('zonal_mean_QT_flux')
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        Stats.write_surface_zonal_mean('zonal_mean_U_flux', np.mean(self.U_flux, axis=1))
        Stats.write_surface_zonal_mean('zonal_mean_V_flux', np.mean(self.V_flux, axis=1))
        Stats.write_surface_zonal_mean('zonal_mean_T_flux', np.mean(self.T_flux, axis=1))
        Stats.write_surface_zonal_mean('zonal_mean_QT_flux', np.mean(self.QT_flux, axis=1))
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_2D_variable(Pr, TS.t,  'U_flux',  self.U_flux)
        Stats.write_2D_variable(Pr, TS.t,  'V_flux',  self.V_flux)
        Stats.write_2D_variable(Pr, TS.t,  'T_flux',  self.T_flux)
        Stats.write_2D_variable(Pr, TS.t,  'QT_flux', self.QT_flux)
        return
