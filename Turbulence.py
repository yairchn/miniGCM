
from Grid import Grid
from DiagnosticVariables import DiagnosticVariables
import numpy as np
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping
from Parameters import Parameters

def TurbulenceFactory(namelist):
    if namelist['turbulence']['turbulence_model'] == 'None':
        return TurbulenceNone(namelist)
    elif namelist['turbulence']['turbulence_model'] == 'DownGradient':
        return DownGradientTurbulence(namelist)
    else:
        print('case not recognized')
    return

class TurbulenceBase:
    def __init__(self, namelist):
        return
    def initialize(self, Pr, namelist):
        return
    def update(self, Pr, Gr, PV, DV):
        return
    def initialize_io(self, Stats):
        return
    def stats_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class TurbulenceNone(TurbulenceBase):
    def __init__(self, namelist):
        TurbulenceBase.__init__(self, namelist)
        return
    def initialize(self, Pr, namelist):
        return
    def update(self, Pr, Gr, PV, DV):
        return
    def initialize_io(self, Stats):
        return
    def stats_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class DownGradientTurbulence(TurbulenceBase):
    def __init__(self, namelist):
        TurbulenceBase.__init__(self, namelist)
        return
    def initialize(self, Pr, namelist):
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers

        Pr.Pstrato = namelist['turbulence']['stratospheric_pressure']
        Pr.Ppbl = namelist['turbulence']['boundary_layer_top_pressure']
        Pr.Dh = namelist['turbulence']['sensible_heat_transfer_coeff']
        Pr.Dq = namelist['turbulence']['latent_heat_transfer_coeff']
        return

    def update(self, Pr, Gr, PV, DV):
        nx = Pr.nlats
        ny = Pr.nlons
        nl = Pr.n_layers
        for i in range(nx):
            for j in range(ny):
                windspeed = sqrt(DV.U.values[i,j,nl-1]*DV.U.values[i,j,nl-1] + DV.V.values[i,j,nl-1]*DV.V.values[i,j,nl-1])
                za = gz[i,j,nl-1]/Pr.g/2.0
                for k in range(nl):
                    if k==0:
                        PV.T.TurbFlux[i,j,k] =  0.0
                        PV.QT.TurbFlux[i,j,k] =  0.0
                    else:
                        # Ke is on pressure levels
                        # eq. (17) Tatcher and Jablonowski 2016
                        Kq = Pr.Dq*windspeed*za*np.exp(-np.power( (np.max(Ppbl - PV.P.values[i,j,k],0.0)/Pr.Pstrato),   2.0))
                        Kh = Pr.Dh*windspeed*za*np.exp(-np.power( (np.max(Ppbl - PV.P.values[i,j,k],0.0)/Pr.Pstrato),   2.0))
                        Th_dn = PV.T.values[i,j,k]  *pow((PV.P.values[i,j,k]   + PV.P.values[i,j,k+1])/2.0/Pr.p_ref, Pr.kappa)
                        Th_up = PV.T.values[i,j,k-1]*pow((PV.P.values[i,j,k-1] + PV.P.values[i,j,k])/2.0/Pr.p_ref, Pr.kappa)
                        dpi = 2.0/(p[ijkp+1]-p[ijkp-1]); # pressure differnece from mid-layer values for ijk
                        PV.T.TurbFlux[i,j,k] = -Kh*(Th_dn - Th_up)*dpi
                        PV.QT.TurbFlux[i,j,k] = -Kq*(PV.QT.values[i,j,k] - PV.QT.values[i,j,k-1])*dpi
        return

    def initialize_io(self, Stats):
        # Stats.add_zonal_mean('zonal_mean_QT_Turb')
        # Stats.add_zonal_mean('zonal_mean_T_Turb')
        return

    def stats_io(self, Stats):
        # Stats.write_zonal_mean('zonal_mean_QT_Turb', PV.QT.TurbFlux)
        # Stats.write_zonal_mean('zonal_mean_T_Turb', PV.T.TurbFlux)
        return

    def io(self, Pr, TS, Stats):
        # Stats.write_3D_variable(Pr, TS.t,  'QT_Turb',  PV.QT.TurbFlux)
        # Stats.write_3D_variable(Pr, TS.t,  'T_Turb',PV.T.TurbFlux)
        return
