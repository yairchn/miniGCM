import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from Parameters import Parameters
from TimeStepping import TimeStepping
from PrognosticVariables import PrognosticVariables
from DiagnosticVariables import DiagnosticVariables
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats

def MicrophysicsFactory(namelist):
    if namelist['microphysics']['microphysics_model'] == 'None':
        return MicrophysicsNone(namelist)
    elif namelist['microphysics']['microphysics_model'] == 'Cutoff':
        if namelist['thermodynamics']['thermodynamics_type'] == 'dry':
             sys.exit('Cannot run microphysics with dry thermodynamics')
        return MicrophysicsCutoff(namelist)
    else:
        print('case not recognized')
    return

class MicrophysicsBase:
    def __init__(self, namelist):
        return
    def initialize(self, Pr, PV, DV, namelist):
        return
    def update(self, Pr, PV, DV, TS):
        return
    def initialize_io(self, Stats):
        return
    def stats_io(self, Gr, PV, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class MicrophysicsNone(MicrophysicsBase):
    def __init__(self, namelist):
        MicrophysicsBase.__init__(self, namelist)
        return
    def initialize(self, Pr, PV, DV, namelist):
        PV.QT.mp_tendency = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.T.mp_tendency  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        return
    def update(self, Pr, PV, DV, TS):
        return
    def initialize_io(self, Stats):
        return
    def stats_io(self, Gr, PV, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class MicrophysicsCutoff(MicrophysicsBase):
    def __init__(self, namelist):
        MicrophysicsBase.__init__(self, namelist)
        return

    def initialize(self, Pr, PV, DV, namelist):
        nl = Pr.n_layers
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.qv_star  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        Pr.max_ss    =  namelist['microphysics']['max_supersaturation']
        Pr.rho_w     =  namelist['microphysics']['water_density']
        Pr.mp_dt     =  namelist['microphysics']['autoconversion_timescale']
        for k in range(nl):
            P_half = np.multiply(np.add(PV.P.values[:,:,k],PV.P.values[:,:,k+1]),0.5)
            # Clausius–Clapeyron equation based saturation
            qv_star = np.multiply(np.divide(np.multiply(Pr.pv_star0,Pr.eps_v),P_half),
                np.exp(-np.multiply(np.divide(Pr.Lv,Pr.Rv),np.subtract(np.divide(1.0,PV.T.values[:,:,k]),np.divide(1.0,Pr.T_0)))))

            DV.QL.values.base[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
        return

    def initialize_io(self, Stats):
        Stats.add_global_mean('global_mean_dQTdt')
        Stats.add_surface_global_mean('global_mean_RainRate')
        Stats.add_zonal_mean('zonal_mean_dQTdt')
        Stats.add_meridional_mean('meridional_mean_dQTdt')
        Stats.add_surface_zonal_mean('zonal_mean_RainRate')
        Stats.add_global_mean('global_mean_qv_star')
        Stats.add_zonal_mean('zonal_mean_qv_star')
        Stats.add_meridional_mean('meridional_mean_qv_star')
        return

    def update(self, Pr, PV, DV, TS):
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers
        Lv_Rv = Pr.Lv/Pr.Rv
        Lv_cpRv = np.pow(Pr.Lv,2.0)/Pr.cp/Pr.Rv
        Lv_cp = Pr.Lv / Pr.cp
        e0_epsv = Pr.e0 * Pr.eps_v
        T_0_inv = 1.0/Pr.T_0

        if (TS.t%Pr.mp_dt == 0.0):
            for i in range(nx):
                for j in range(ny):
                    for k in range(nl):
                        p_half = 0.5*(PV.P.values[i,j,k] + PV.P.values[i,j,k+1])
                        qv_star[ijk] = (e0_epsv/p_half)*exp(-Lv_Rv*(1.0/T[ijk]-T_0_inv)) # Eq. (1) Tatcher and Jabolonski 2016
                        denom = (1.0+Lv_cpRv*self.qv_star[i,j,k]/(PV.T.values[i,j,k] * PV.T.values[i,j,k]))*Pr.mp_dt
                        DV.QL.values[i,j,k] = PV.QT.values[ijk] - self.qv_star[i,j,k]
                        PV.T.mp_tendency[ijk]  =  (PV.QT.values[i,j,k] - self.qv_star[i,j,k])/denom # Eq. (2) Tatcher and Jabolonski 2016
                        PV.QT.mp_tendency[i,j,k] = -(PV.QT.values[i,j,k] - (1.0+Pr.max_ss)*self.qv_star[i,j,k])/denom # Eq. (3) Tatcher and Jabolonski 2016
                        self.RainRate[i,j] = self.RainRate[i,j] - (PV.QT.mp_tendency[i,j,k]/Pr.rho_w*Pr.g*(PV.P.values[i,j,k] - PV.P.values[0,0,0]))/(nl) # Eq. (5) Tatcher and Jabolonski 2016

        else:
            PV.QT.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            PV.T.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            self.RainRate = np.zeros((nx,ny),dtype=np.float64, order='c')
            self.qv_star = np.zeros((nx,ny,nl),dtype=np.float64, order='c')


        return

    def stats_io(self, Gr, PV, Stats):
        Stats.write_global_mean(Gr, 'global_mean_dQTdt', PV.QT.mp_tendency)
        Stats.write_surface_global_mean(Gr, 'global_mean_RainRate', self.RainRate)
        Stats.write_zonal_mean('zonal_mean_dQTdt',PV.QT.mp_tendency)
        Stats.write_meridional_mean('meridional_mean_dQTdt',PV.QT.mp_tendency)
        Stats.write_surface_zonal_mean('zonal_mean_RainRate',self.RainRate)
        Stats.write_global_mean(Gr, 'global_mean_qv_star',self.qv_star)
        Stats.write_zonal_mean('zonal_mean_qv_star',self.qv_star)
        Stats.write_meridional_mean('meridional_mean_qv_star',self.qv_star)
        return

    def io(self, Pr, TS, Stats):
        Stats.write_2D_variable(Pr, int(TS.t) , 'Rain_Rate',self.RainRate)
        return
