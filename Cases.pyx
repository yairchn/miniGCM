import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
cimport Forcing
cimport Surface
cimport Microphysics
import sys
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters
import time
import sphericalForcing as spf

def CasesFactory(namelist):
    if namelist['meta']['casename'] == 'HeldSuarez':
        return HeldSuarez(namelist)
    elif namelist['meta']['casename'] == 'HeldSuarezMoist':
        return HeldSuarezMoist(namelist)
    # anthoer example
    # elif namelist['meta']['casename'] == 'Stochastic_Forcing':
    #     return Stochastic_Frorcing(paramlist)
    else:
        print('case not recognized')
    return



cdef class CaseBase:
    def __init__(self, namelist):
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        return

    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        return

    cpdef initialize_forcing(self, Parameters Pr, Grid Gr, namelist):
        return

    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV,namelist):
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return

    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return

cdef class MoistVortex(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.ForcingBettsMiller()
        self.Sur = Surface.SurfaceNone()
        self.MP = Microphysics.MicrophysicsNone()
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        cdef:
            double [:,:] noise

        PV.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        PV.T_init        = np.array([229.0, 257.0, 295.0])

        Pr.sigma_b = namelist['forcing']['sigma_b']
        Pr.k_a = namelist['forcing']['k_a']
        Pr.k_b = namelist['forcing']['k_b']
        Pr.k_s = namelist['forcing']['k_s']
        Pr.k_f = namelist['forcing']['k_f']
        Pr.DT_y = namelist['forcing']['equator_to_pole_dT']
        Pr.Dtheta_z = namelist['forcing']['lapse_rate']
        Pr.T_equator = namelist['forcing']['equatorial_temperature']

        PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.QT.values         = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c')
        PV.P.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),PV.P_init)
        PV.T.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),PV.T_init)

        if Pr.inoise==1: # load the random noise to grid space
             noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')/10.0
             PV.T.values.base[:,:,Pr.n_layers-1] = np.add(PV.T.values.base[:,:,Pr.n_layers-1],noise.base)
        print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
        if Pr.inoise==1:
             # load the random noise to grid space
             #noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')*Pr.noise_amp
             # calculate noise
             spec_zeros = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
             fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0.)
             noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(spec_zeros))*Pr.noise_amp
             # add noise
             PV.T.spectral.base[:,Pr.n_layers-1] = np.add(PV.T.spectral.base[:,Pr.n_layers-1],
                                                        Gr.SphericalGrid.grdtospec(noise.base))
        print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
        return

    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        self.Sur.initialize(Pr, Gr, PV, namelist)
        return

    cpdef initialize_forcing(self, Parameters Pr, Grid Gr, namelist):
        self.Fo.initialize(Pr, Gr, namelist)
        return

    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV,namelist):
        self.MP.initialize(Pr, PV, DV, namelist)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CaseBase.initialize_io(self, Stats)
        self.Fo.initialize_io(Stats)
        self.Sur.initialize_io(Stats)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        CaseBase.io(self, Pr, TS, Stats)
        self.Fo.io(Pr, TS, Stats)
        self.Sur.io(Pr, TS, Stats)
        return

    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        CaseBase.stats_io(self, PV, Stats)
        self.Fo.stats_io(Stats)
        self.Sur.stats_io(Stats)
        return

    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        self.Sur.update(Pr, Gr, PV, DV)
        self.Fo.update(Pr, Gr, PV, DV)
        self.MP.update(Pr, PV, DV, TS)
        return