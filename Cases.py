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


def CasesFactory(namelist, Gr):
    if namelist['meta']['casename'] == 'HeldSuarez':
        return HeldSuarez(namelist, Gr)
    elif namelist['meta']['casename'] == 'HeldSuarez_moist':
        return HeldSuarez_moist(namelist, Gr)
    # anthoer example
    # elif namelist['meta']['casename'] == 'Stochastic_Forcing':
    #     return Stochastic_Frorcing(paramlist)
    else:
        print('case not recognized')
    return


class CasesBase:
    def __init__(self, namelist):
        return
    # def initialize_prognostic_vars(self, Gr, PV, DV):
    #     return
    def initialize_surface(self, Gr):
        return
    def initialize_forcing(self, Gr, PV, DV, namelist):
        return
    def initialize_io(self, Stats):
        return
    def io(self, PV, Gr, TS, Stats):
        Stats.write_2D_variable(Gr, int(TS.t), 'shf', PV.T.SurfaceFlux)
        return
    def update_surface(self):
        return
    def update_forcing(self):
        return

class HeldSuarez(CasesBase):
    def __init__(self, namelist, Gr):
        self.casename = 'HelzSuarez'
        self.tau = 50.0*24.0*3600.0
        self.dTdy        = 60.0
        self.T_lapserate = 10.0
        self.p_ref       = 10e5
        self.foring_type = 'relaxation'
        #Temperature profiles from Held and Suarez (1994) -- initialize isothermally in each layer
        self.Tinit  = [229.0, 257.0, 295.0] # make this a matrix size self.T.values 
        self.Fo  = Forcing.Forcing_HelzSuarez()
        self.Sur = Surface.SurfaceNone()
        self.RS = ReferenceState.ReferenceState()
        self.DV = DiagnosticVariables(Gr)
        self.PV = PrognosticVariables(Gr)

        return

    def initialize(self, Gr, PV, DV):
        self.Base_pressure = 100000.0
        self.P_init        = [Gr.p1, Gr.p2, Gr.p3, Gr.p_ref]
        self.T_init        = [229.0, 257.0, 295.0]
        self.QT_init       = [0.0, 0.0, 0.0]

        PV.Vorticity.values  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        PV.P.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers+1), dtype=np.double, order='c'),self.P_init)
        PV.T.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),   dtype=np.double, order='c'),self.T_init)
        PV.QT.values         = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),   dtype=np.double, order='c'),self.QT_init)
        # initilize spectral values
        for k in range(Gr.n_layers):
            PV.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,k])
            PV.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values[:,:,k])
            PV.QT.spectral[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values[:,:,k])
            PV.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values[:,:,k])
            PV.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values[:,:,k])
        PV.P.spectral[:,Gr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,Gr.n_layers])

        return

    def initialize_surface(self, Gr, namelist):
        self.Sur.initialize()
        return

    def initialize_forcing(self, Gr, PV, DV, namelist):
        self.Fo.initialize(Gr, PV, DV, namelist)
        return

    def initialize_io(self, Stats):
        Stats.initialize_io(self, Stats)
        return

    def io(self, PV, Gr, TS, Stats):
        CasesBase.io(self, PV, Gr, TS, Stats)
        return

    def update_reference(self, Gr):
        # self.RS.update(Gr, Stats)
        return

    def update_surface(self, Gr):
        # self.Sur.update(self)
        return

    def update_forcing(self, TS, Gr, PV, DV, namelist):
        self.Fo.update(TS, Gr, PV, DV, namelist)
        return


class HeldSuarez_moist(CasesBase):
    def __init__(self, namelist, Gr):
        self.casename = 'HelzSuarez'
        self.tau = 50.0*24.0*3600.0
        self.dTdy        = 60.0
        self.T_lapserate = 10.0
        self.p_ref       = 10e5
        self.foring_type = 'relaxation'
        #Temperature profiles from Held and Suarez (1994) -- initialize isothermally in each layer
        self.Tinit  = [229.0, 257.0, 295.0] # make this a matrix size self.T.values 
        self.Fo  = Forcing.Forcing_HelzSuarez()
        self.Sur = Surface.SurfaceNone()
        self.RS = ReferenceState.ReferenceState()
        self.DV = DiagnosticVariables(Gr)
        self.PV = PrognosticVariables(Gr)

        return

    def initialize(self, Gr, PV, DV):
        self.Base_pressure = 100000.0
        self.P_init        = [Gr.p1, Gr.p2, Gr.p3, Gr.p_ref]
        self.T_init        = [229.0, 257.0, 295.0]
        self.QT_0       = 18.0/1000.0

        PV.Vorticity.values  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.double, order='c')
        PV.P.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers+1), dtype=np.double, order='c'),self.P_init)
        PV.T.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),   dtype=np.double, order='c'),self.T_init)
        for k in range(Gr.n_layers):
            sigma = PV.P.values[:,:,k]/PV.P.values[:,:,Gr.n_layers]
            PV.QT.values[:,:,k] = (self.QT_0*np.exp(-(Gr.lat[:,0]/Gr.phi_hw)**4.0)
                *np.exp(-((sigma-1)*(PV.P.values[:,:,k]/Gr.P_hw))**2.0))

        # initilize spectral values
        for k in range(Gr.n_layers):
            PV.P.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,k])
            PV.T.spectral[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values[:,:,k])
            PV.QT.spectral[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values[:,:,k])
            PV.Vorticity.spectral[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values[:,:,k])
            PV.Divergence.spectral[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values[:,:,k])
        PV.P.spectral[:,Gr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,Gr.n_layers])

        return

    def initialize_surface(self, Gr, namelist):
        self.Sur.initialize()
        return

    def initialize_forcing(self, Gr, PV, DV, namelist):
        self.Fo.initialize(Gr, PV, DV, namelist)
        return

    def initialize_io(self, Stats):
        Stats.initialize_io(self, Stats)
        return

    def io(self, PV, Gr, TS, Stats):
        CasesBase.io(self, PV, Gr, TS, Stats)
        return

    def update_reference(self, Gr):
        # self.RS.update(Gr, Stats)
        return

    def update_surface(self, Gr):
        # self.Sur.update(self)
        return

    def update_forcing(self, TS, Gr, PV, DV, namelist):
        self.Fo.update(TS, Gr, PV, DV, namelist)
        return
