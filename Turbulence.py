
from Grid import Grid
from DiagnosticVariables import DiagnosticVariables
import numpy as np
from NetCDFIO import NetCDFIO_Stats
from PrognosticVariables import PrognosticVariables
from TimeStepping import TimeStepping
from Parameters import Parameters


# def extern from "turbulence_functions.h":
#     void vertical_turbulent_flux(double g, double Ch, double Cq, double kappa, double p_ref,
#                                  double Ppbl, double Pstrato, double* p, double* gz,
#                                  double* T, double* qt, double* u, double* v, double* wTh,
#                                  double* wqt, Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

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

        # vertical_turbulent_flux(Pr.g, Pr.Dh, Pr.Dq, Pr.kappa, Pr.p_ref, Pr.Ppbl, Pr.Pstrato,
        #                     &PV.P.values[0,0,0],&DV.gZ.values[0,0,0],
        #                     &PV.T.values[0,0,0],&PV.QT.values[0,0,0],
        #                     &DV.U.values[0,0,0],&DV.V.values[0,0,0],
        #                     &PV.T.TurbFlux[0,0,0],&PV.QT.TurbFlux[0,0,0],
        #                     nx, ny, nl)
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
