import numpy as np
from Grid import Grid
from DiagnosticVariables import DiagnosticVariables
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping
from Parameters import Parameters

def SurfaceFactory(namelist):
    if namelist['surface']['surface_model'] == 'None':
        return SurfaceNone(namelist)
    elif namelist['surface']['surface_model'] == 'BulkFormula':
        return SurfaceBulkFormula(namelist)
    else:
        print('case not recognized')
    return

class SurfaceBase:
    def __init__(self, namelist):
        return
    def initialize(self, Pr, Gr, PV, namelist):
        return
    def update(self, Pr, Gr, PV, DV):
        return
    def initialize_io(self, Stats):
        return
    def stats_io(self, Gr, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class SurfaceNone(SurfaceBase):
    def __init__(self, namelist):
        SurfaceBase.__init__(self, namelist)
        return
    def initialize(self, Pr, Gr, PV, namelist):
        return
    def update(self, Pr, Gr, PV, DV):
        return
    def initialize_io(self, Stats):
        return
    def stats_io(self, Gr, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class SurfaceBulkFormula(SurfaceBase):
    def __init__(self, namelist):
        SurfaceBase.__init__(self, namelist)
        return
    def initialize(self, Pr, Gr, PV, namelist):
        Pr.Cd = namelist['surface']['momentum_transfer_coeff']
        Pr.Ch = namelist['surface']['sensible_heat_transfer_coeff']
        Pr.Cq = namelist['surface']['latent_heat_transfer_coeff']
        Pr.dT_s = namelist['surface']['surface_temp_diff']
        Pr.T_min = namelist['surface']['surface_temp_min']
        Pr.dphi_s = namelist['surface']['surface_temp_lat_dif']
        # eq. (6) Tatcher and Jablonowski 2016
        self.T_surf  = np.multiply(Pr.dT_s,np.exp(-0.5*np.power(Gr.lat,2.0)/Pr.dphi_s**2.0)) + Pr.T_min
        self.QT_surf = np.multiply(np.divide(Pr.pv_star0*Pr.eps_v, PV.P.values[:,:,Pr.n_layers]),
                       np.exp(-np.multiply(Pr.Lv/Pr.Rv,np.subtract(np.divide(1,self.T_surf) , np.divide(1,Pr.T_0) ))))
        return

    def update(self, Pr, Gr, PV, DV):
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers
        T_0_inv = 1/Pr.T_0
        Lv_Rv=Pr.Lv / Pr.Rv
        pv0epsv=Pr.pv_star0*Pr.eps_v

        windspeed = np.sqrt(DV.U.values[:,:,nl-1]*DV.U.values[:,:,nl-1] +
                         DV.V.values[:,:,nl-1]*DV.V.values[:,:,nl-1])
        qt_surf = np.multiply(np.divide(pv0epsv, PV.P.values[:,:,nl]),
                    np.exp(-np.multiply(Lv_Rv,(np.subtract(np.divide(1.0,self.T_surf), T_0_inv)))))
        z_a = np.divide(DV.gZ.values[:,:,nl-1], Pr.g * 2.0)
        u_surf_flux  = np.multiply(np.multiply(np.divide(-Pr.Cd / z_a),windspeed),DV.U.values[:,:,nl-1])
        v_surf_flux  = np.multiply(np.multiply(np.divide(-Pr.Cd / z_a),windspeed),DV.V.values[:,:,nl-1])
        T_surf_flux  = np.multiply(np.multiply(np.divide(-Pr.Ch / z_a),windspeed),np.subtract(PV.T.values[:,:,nl-1], self.T_surf))
        qt_surf_flux = np.multiply(np.multiply(np.divide(-Pr.Cq / z_a),windspeed),np.subtract(PV.QT.values[:,:,nl-1], qt_surf))
        return
    def initialize_io(self, Stats):
        Stats.add_surface_zonal_mean('zonal_mean_T_surf')
        Stats.add_surface_zonal_mean('zonal_mean_QT_surf')
        return

    def stats_io(self, Gr, Stats):
        Stats.write_surface_zonal_mean('zonal_mean_T_surf', self.T_surf)
        Stats.write_surface_zonal_mean('zonal_mean_QT_surf', self.QT_surf)
        Stats.write_surface_global_mean(Gr, 'global_mean_T_surf', self.T_surf)
        Stats.write_surface_global_mean(Gr, 'global_mean_QT_surf', self.QT_surf)
        return

    def io(self, Pr, TS, Stats):
        Stats.write_2D_variable(Pr, TS.t,  'T_surf', self.T_surf)
        Stats.write_2D_variable(Pr, TS.t,  'QT_surf', self.QT_surf)
        return
