import numpy as np
from NetCDFIO import Stats
from PrognosticVariables import PrognosticVariables
from DiagnosticVariables import DiagnosticVariables
from Grid import Grid
from NetCDFIO import Stats
import TimeStepping
import Forcing
import Surface
import ReferenceState
import pylab as plt
import Parameters
import Microphysics


def CasesFactory(Pr, namelist):
    if namelist['meta']['casename'] == 'HeldSuarez':
        return HeldSuarez(Pr, namelist)
    elif namelist['meta']['casename'] == 'HeldSuarez_moist':
        return HeldSuarez_moist(Pr, namelist)
    # anthoer example
    # elif namelist['meta']['casename'] == 'Stochastic_Forcing':
    #     return Stochastic_Frorcing(paramlist)
    else:
        print('case not recognized')
    return


class CasesBase:
    def __init__(self, Pr, namelist):
        return
    def initialize(self, Pr, Gr, PV, namelist):
        return
    def initialize_surface(self, Pr, Gr, PV):
        return
    def initialize_forcing(self, Pr):
        return
    def initialize_microphysics(self, Pr, namelist):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, PV, Stats):
        return
    def update_surface(self, Pr, Gr):
        return
    def update_forcing(self, Pr, Gr, TS,  PV, DV):
        return
    def update_microphysics(self, Pr, Gr, PV, TS):
        return

class HeldSuarez(CasesBase):
    def __init__(self, Pr, namelist):
        Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.Forcing_HelzSuarez()
        self.Sur = Surface.SurfaceNone()
        self.MP = Microphysics.MicrophysicsNone()
        return

    def initialize(self, Pr, Gr, PV, namelist):
        PV.P_init        = [Pr.p1, Pr.p2, Pr.p3, Pr.p_ref]
        PV.T_init        = [229.0, 257.0, 295.0]

        Pr.sigma_b = namelist['forcing']['sigma_b']
        Pr.k_a = namelist['forcing']['k_a']
        Pr.k_s = namelist['forcing']['k_s']
        Pr.k_f = namelist['forcing']['k_f']
        Pr.DT_y = namelist['forcing']['equator_to_pole_dT']
        Pr.T_equator = namelist['forcing']['equatorial_temperature']
        Pr.Dtheta_z = namelist['forcing']['lapse_rate']
        Pr.Tbar0 = namelist['forcing']['relaxation_temperature']
        Pr.cp = namelist['thermodynamics']['heat_capacity']
        Pr.Rd = namelist['thermodynamics']['dry_air_gas_constant']


        PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.QT.values         = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c')
        PV.P.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),PV.P_init)
        PV.T.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),PV.T_init)
        # initilize spectral values
        for k in range(Pr.n_layers):
            PV.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,k])
            PV.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values[:,:,k])
            PV.QT.spectral[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values[:,:,k])
            PV.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values[:,:,k])
            PV.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values[:,:,k])
        PV.P.spectral[:,Pr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,Pr.n_layers])

        return

    def initialize_surface(self, Pr, Gr, PV):
        return

    def initialize_forcing(self, Pr):
        self.Fo.initialize(Pr)
        return

    def initialize_microphysics(self, Pr, namelist):
        self.MP.initialize(Pr, namelist)
        return

    def initialize_io(self, Stats):
        Stats.initialize_io(self, Stats)
        return

    def io(self, Pr, TS, PV, Stats):
        CasesBase.io(self, Pr, PV, TS, Stats)
        return

    def update_surface(self, Pr, Gr, TS, PV):
        self.Sur.update(Pr, Gr, TS, PV)
        return

    def update_forcing(self, Pr, Gr, TS,  PV, DV):
        self.Fo.update(Pr, Gr, TS, PV, DV)
        return

    def update_microphysics(self, Pr, Gr, PV, TS):
        return


class HeldSuarez_moist(CasesBase):
    def __init__(self, Pr, namelist):
        Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.Forcing_HelzSuarezMoist()
        self.Sur = Surface.Surface_BulkFormula()
        self.MP = Microphysics.MicrophysicsCutoff()
        return

    def initialize(self, Pr, Gr, PV, namelist):
        PV.QT_0      = namelist['forcing']['initial_surface_qt']
        Pr.T_0       = namelist['thermodynamics']['triple_point_temp']
        Pr.P_hw      = namelist['thermodynamics']['verical_half_width_of_the_q']
        Pr.phi_hw    = namelist['thermodynamics']['horizontal_half_width_of_the_q']
        Pr.sigma_b   = namelist['forcing']['sigma_b']
        Pr.k_a       = namelist['forcing']['k_a']
        Pr.k_s       = namelist['forcing']['k_s']
        Pr.k_f       = namelist['forcing']['k_f']
        Pr.T_equator = namelist['forcing']['equatorial_temperature']
        Pr.T_pole    = namelist['forcing']['polar_temperature']
        Pr.init_k    = namelist['forcing']['initial_profile_power']
        Pr.DT_y      = namelist['forcing']['equator_to_pole_dT']
        Pr.Dtheta_z  = namelist['forcing']['lapse_rate']
        Pr.Tbar0     = namelist['forcing']['relaxation_temperature']
        Gamma        = namelist['forcing']['Gamma_init']
        z            = np.linspace(0,20000,200)

        PV.P_init            = [Pr.p1, Pr.p2, Pr.p3, Pr.p_ref]
        PV.T_init            = [245.0, 282.0, 303.0]
        Tv_                  = np.zeros((len(Gr.lat[:,0]), len(z), Pr.n_layers),  dtype=np.double, order='c')
        QT_                  = np.zeros((len(Gr.lat[:,0]), len(z), Pr.n_layers),  dtype=np.double, order='c')
        qv_star_             = np.zeros((len(Gr.lat[:,0]), len(z), Pr.n_layers),  dtype=np.double, order='c')
        p                    = np.zeros((len(Gr.lat[:,0]), len(z)),  dtype=np.double, order='c')
        PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.P.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),PV.P_init)

        T_0 = 0.5 * (Pr.T_equator + Pr.T_pole)
        B   = (T_0 - Pr.T_pole) / T_0 / Pr.T_pole
        C   = 0.5 * (Pr.init_k + 2.0)*(Pr.T_equator-Pr.T_pole)/Pr.T_equator/Pr.T_pole

        tau_z_1   = np.exp(Gamma * z / T_0)
        tau_z_2   = 1.0 - np.multiply(2.0, np.power(np.divide(z / 2.0, Pr.Rd * T_0 / Pr.g),2.0))
        tau_z_3   = np.exp(-np.power(np.divide(z / 2.0 , Pr.Rd * T_0 / Pr.g),2.0))
        tau_1     = np.multiply(np.divide(1.0 , T_0) , tau_z_1) + np.multiply(np.multiply(B , tau_z_2) , tau_z_3)
        tau_2     = np.multiply(np.multiply(C , tau_z_2) , tau_z_3)
        tau_int_1 = np.multiply(1.0 / Gamma , (tau_z_1 - 1.0)) + np.multiply(np.multiply(B , z ), tau_z_3)
        tau_int_2 = np.multiply(np.multiply(C , z) ,tau_z_3);
        I_T       = (np.power(np.cos(Gr.lat[:,0]),Pr.init_k)
                   - Pr.init_k/(Pr.init_k + 2.0) * np.power(np.cos(Gr.lat[:,0]),Pr.init_k+2.0))

        for i in range(len(Gr.lat[:,0])):
            for j in range(len(z)):
                p[i,j] = Pr.p_ref * np.exp(-Pr.g / Pr.Rd * (tau_int_1[j] - tau_int_2[j] * I_T[i]))
                for k in range(Pr.n_layers):
                    if (p[i,j]>PV.P_init[k] and p[i,j]<PV.P_init[k+1]):
                        Tv_[i,j,k] = np.power(tau_1[j] - tau_2[j] * I_T[i],-1.0)
                        QT_[i,j,k] = (PV.QT_0 * np.exp(-(Gr.lat[i,0] / Pr.phi_hw)**4.0)
                                    * np.exp(-((p[i,j]/Pr.p_ref - 1.0) * Pr.p_ref / Pr.P_hw)**2.0))
                    else:
                        Tv_[i,j,k] = np.nan
                        QT_[i,j,k] = np.nan

        for k in range(Pr.n_layers):
            QT_meridional = (PV.QT_0 * np.exp(-(Gr.lat[:,0] / Pr.phi_hw)**4.0)
                * np.exp(-((PV.P_init[k]/Pr.p_ref - 1.0) * Pr.p_ref / Pr.P_hw)**2.0))
            # QT_meridional = np.nanmean(QT_[:,:,k], axis=1)
            eps = 1.0/Pr.eps_v-1.0
            T_meridional = np.nanmean(Tv_[:,:,k]/(1+0.608*QT_[:,:,k]), axis=1)
            PV.QT.values[:,:,k] = np.repeat(QT_meridional[:, np.newaxis], Pr.nlons, axis=1)
            PV.T.values[:,:,k]  = np.repeat(T_meridional[:, np.newaxis], Pr.nlons, axis=1)

        # # initilize spectral values
        for k in range(Pr.n_layers):
            PV.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,k])
            PV.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values[:,:,k])
            PV.QT.spectral[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values[:,:,k])
            PV.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values[:,:,k])
            PV.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values[:,:,k])
        PV.P.spectral[:,Pr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,Pr.n_layers])

        return

    def initialize_surface(self, Pr, Gr, PV):
        self.Sur.initialize(Pr, Gr, PV)
        return

    def initialize_forcing(self, Pr):
        self.Fo.initialize(Pr)
        return

    def initialize_microphysics(self, Pr, PV, namelist):
        self.MP.initialize(Pr, PV, namelist)
        return

    def initialize_io(self, Stats):
        Stats.initialize_io(self, Stats)
        return

    def io(self, Pr, TS, PV, Stats):
        CasesBase.io(self, Pr, TS, PV, Stats)
        return

    def update_surface(self, Pr, Gr, TS, PV, DV):
        # self.Sur.update(Pr, Gr, TS, PV, DV)
        return

    def update_forcing(self, Pr, Gr, TS,  PV, DV):
        self.Fo.update(Pr, Gr, TS, PV, DV)
        return

    def update_microphysics(self, Pr, Gr, PV, TS):
        self.MP.update(Pr, Gr, TS, PV)
        return
