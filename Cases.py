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
    def initialize(self, Pr, Gr, PV):
        return
    def initialize_surface(self, Pr, Gr):
        return
    def initialize_forcing(self, Pr, Gr, PV, DV):
        return
    def initialize_microphysics(self, Pr, Gr, PV, DV):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, Gr, TS, PV, Stats):
        return
    def update_surface(self, Pr, Gr):
        self.Sur.update()
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

    def initialize(self, Pr, Gr, PV):
        self.Base_pressure = 100000.0
        PV.P_init        = [Pr.p1, Pr.p2, Pr.p3, Pr.p_ref]
        PV.T_init        = [229.0, 257.0, 295.0]
        self.QT_init       = [0.0, 0.0, 0.0]

        Pr.sigma_b = namelist['forcing']['sigma_b']
        Pr.k_a = namelist['forcing']['k_a']
        Pr.k_s = namelist['forcing']['k_s']
        Pr.k_f = namelist['forcing']['k_f']
        Pr.DT_y = namelist['forcing']['equator_to_pole_dT']
        Pr.T_equator = namelist['thermodynamics']['equatorial Temperature']
        Pr.Dtheta_z = namelist['forcing']['lapse_rate']
        Pr.Tbar0 = namelist['forcing']['relaxation_temperature']
        Pr.cp = namelist['thermodynamics']['heat_capacity']
        Pr.Rd = namelist['thermodynamics']['dry_air_gas_constant']


        PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.P.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),PV.P_init)
        PV.T.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),PV.T_init)
        PV.QT.values         = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),self.QT_init)
        # initilize spectral values
        for k in range(Pr.n_layers):
            PV.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,k])
            PV.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values[:,:,k])
            PV.QT.spectral[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values[:,:,k])
            PV.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values[:,:,k])
            PV.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values[:,:,k])
        PV.P.spectral[:,Pr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,Pr.n_layers])

        return

    def initialize_surface(self, Pr, Gr):
        return

    def initialize_forcing(self, Pr, Gr, PV, DV):
        self.Fo.initialize(Gr, PV, DV, namelist)
        return

    def initialize_microphysics(self, Pr, Gr, PV, DV):
        return

    def initialize_io(self, Stats):
        Stats.initialize_io(self, Stats)
        return

    def io(self, Pr, Gr, TS, PV, Stats):
        CasesBase.io(self, PV, Gr, TS, Stats)
        return

    def update_surface(self, Pr, Gr):
        self.Sur.update()
        return

    def update_forcing(self, Pr, Gr, TS,  PV, DV):
        self.Fo.update(TS, Gr, PV, DV, namelist)
        return

    def update_microphysics(self, Pr, Gr, PV, TS):
        return


class HeldSuarez_moist(CasesBase):
    def __init__(self, namelist, Pr):
        Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.Forcing_HelzSuarezMoist()
        self.Sur = Surface.Surface_BulkFormula()
        self.MP = Microphysics.MicrophysicsCutoff()
        return

    def initialize(self, Pr, Gr, PV):
        self.Base_pressure = 100000.0
        PV.P_init        = [Pr.p1, Pr.p2, Pr.p3, Pr.p_ref]
        PV.T_init        = [229.0, 257.0, 295.0]
        PV.QT_0       = 18.0/1000.0

        Pr.T_0 = namelist['thermodynamics']['triple_point_temp']
        Pr.P_hw = namelist['thermodynamics']['verical_half_width_of_the_q']
        Pr.phi_hw = namelist['thermodynamics']['horizontal_half_width_of_the_q']
        Pr.sigma_b = namelist['forcing']['sigma_b']
        Pr.k_a = namelist['forcing']['k_a']
        Pr.k_s = namelist['forcing']['k_s']
        Pr.k_f = namelist['forcing']['k_f']
        Pr.DT_y = namelist['forcing']['equator_to_pole_dT']
        Pr.Dtheta_z = namelist['forcing']['lapse_rate']
        Pr.T_equator = namelist['thermodynamics']['equatorial Temperature']
        Pr.Tbar0 = namelist['forcing']['relaxation_temperature']

        PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.P.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),PV.P_init)
        Tv                   = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),PV.T_init)
        for k in range(Pr.n_layers):
            sigma_1 = PV.P_init[k]/PV.P_init[Pr.n_layers]-1.0
            TQ_meridional = np.multiply(PV.QT_0,
                np.exp(-(Gr.lat[:,0]/Gr.phi_hw)**4.0)*np.exp(-(sigma_1*(Pr.p_ref/Gr.P_hw))**2.0))
            PV.QT.values[:,:,k] = np.repeat(TQ_meridional[:, np.newaxis], Pr.nlons, axis=1)
            PV.T.values[:,:,k]  = np.divide(Tv[:,:,k],np.add(1.0,0.608*PV.QT.values[:,:,k]))
        # # initilize spectral values
        for k in range(Pr.n_layers):
            PV.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,k])
            PV.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values[:,:,k])
            PV.QT.spectral[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values[:,:,k])
            PV.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values[:,:,k])
            PV.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values[:,:,k])
        PV.P.spectral[:,Pr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,Pr.n_layers])

        return

    def initialize_surface(self, Pr, Gr, namelist):
        self.Sur.initialize()
        return

    def initialize_forcing(self, Pr, Gr, PV, DV):
        self.Fo.initialize(Gr, PV, DV, namelist)
        return

    def initialize_microphysics(self, Pr, Gr, namelist):
        self.MP.update(TS, Gr, PV, DV, namelist)
        return

    def initialize_io(self, Stats):
        Stats.initialize_io(self, Stats)
        return

    def io(self, Pr, Gr, TS, PV, Stats):
        CasesBase.io(self, PV, Gr, TS, Stats)
        return

    def update_surface(self, Pr, Gr):
        self.Sur.update()
        return

    def update_forcing(self, Pr, Gr, TS,  PV, DV):
        self.Fo.update(TS, Gr, PV, DV, namelist)
        return

    def update_microphysics(self, Pr, Gr, PV, TS):
        self.MP.update(TS, Gr, PV, DV, namelist)
        return
