import numpy as np
import sys
import time
from PrognosticVariables import PrognosticVariables
from DiagnosticVariables import DiagnosticVariables
from Grid import Grid
from NetCDFIO import NetCDFIO_Stats
import Forcing
import Surface
import Microphysics
import Convection
import Turbulence
from TimeStepping import TimeStepping
from Parameters import Parameters
import sphericalForcing as spf
from Restart import Restart

def CasesFactory(namelist):
    if namelist['meta']['casename'] == 'HeldSuarez':
        return HeldSuarez(namelist)
    elif namelist['meta']['casename'] == 'HeldSuarezMoist':
        return HeldSuarezMoist(namelist)
    else:
        print('case not recognized')
    return

class CaseBase:
    def __init__(self, namelist):
        return

    def initialize(self, RS, Pr, Gr, PV, TS, namelist):
        return

    def initialize_surface(self, Pr, Gr, PV, namelist):
        return

    def initialize_convection(self, Pr, Gr, PV, namelist):
        return

    def initialize_turbulence(self, Pr, namelist):
        return

    def initialize_forcing(self, Pr, Gr, namelist):
        return

    def initialize_microphysics(self, Pr, PV, DV,namelist):
        return

    def initialize_io(self, Stats):
        return

    def io(self, Pr, TS, Stats):
        return

    def stats_io(self, Gr, PV, Stats):
        return

    def update(self, Pr, Gr, PV, DV, TS):
        return

class HeldSuarez(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.ForcingFactory(namelist)
        self.Sur = Surface.SurfaceFactory(namelist)
        self.MP  = Microphysics.MicrophysicsFactory(namelist)
        self.Co = Convection.ConvectionFactory(namelist)
        self.Tr = Turbulence.TurbulenceFactory(namelist)
        return

    def initialize(self, RS, Pr, Gr, PV, TS, namelist):
        Pr.sigma_b = namelist['forcing']['sigma_b']
        Pr.k_a = namelist['forcing']['k_a']
        Pr.k_b = namelist['forcing']['k_b']
        Pr.k_s = namelist['forcing']['k_s']
        Pr.k_f = namelist['forcing']['k_f']
        Pr.DT_y = namelist['forcing']['equator_to_pole_dT']
        Pr.Dtheta_z = namelist['forcing']['lapse_rate']
        Pr.T_equator = namelist['forcing']['equatorial_temperature']

        PV.P.values      = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),Pr.pressure_levels)

        if Pr.restart:
            RS.initialize(Pr, Gr, PV, TS, namelist)
        else:
            PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
            PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
            PV.QT.values         = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c')
            PV.T.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),Pr.T_init)

        PV.physical_to_spectral(Pr, Gr)
        if namelist['initialize']['noise']:
             # calculate noise
             F0 = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
             fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0., noise_type=Pr.noise_type)
             noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(F0))*Pr.noise_amp
             # save noise here
             # np.save('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy',noise)
             # load noise here
             # noise = np.load('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy')
             # add noise
             PV.T.spectral[:,Pr.n_layers-1] = np.add(PV.T.spectral[:,Pr.n_layers-1],
                                                        Gr.SphericalGrid.grdtospec(noise))
        print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral[:,Pr.n_layers-1]).min())
        return

    def initialize_surface(self, Pr, Gr, PV, namelist):
        self.Sur.initialize(Pr, Gr, PV, namelist)
        return

    def initialize_turbulence(self, Pr, namelist):
        self.Tr.initialize(Pr, namelist)
        return

    def initialize_convection(self, Pr, Gr, PV, namelist):
        self.Co.initialize(Pr, Gr, namelist)
        return

    def initialize_forcing(self, Pr, Gr, namelist):
        self.Fo.initialize(Pr, Gr, namelist)
        return

    def initialize_microphysics(self, Pr, PV, DV,namelist):
        self.MP.initialize(Pr, PV, DV, namelist)
        return

    def initialize_io(self, Stats):
        CaseBase.initialize_io(self, Stats)
        self.Fo.initialize_io(Stats)
        self.Sur.initialize_io(Stats)
        return

    def io(self, Pr, TS, Stats):
        CaseBase.io(self, Pr, TS, Stats)
        self.Fo.io(Pr, TS, Stats)
        self.Sur.io(Pr, TS, Stats)
        # self.Co.io(Pr, TS, Stats)
        return

    def stats_io(self, Gr, PV, Stats):
        CaseBase.stats_io(self, Gr, PV, Stats)
        self.Fo.stats_io(Stats)
        self.Sur.stats_io(Gr,Stats)
        # self.Co.stats_io(Stats)
        return

    def update(self, Pr, Gr, PV, DV, TS):
        self.Sur.update(Pr, Gr, PV, DV)
        self.Fo.update(Pr, Gr, PV, DV)
        self.Co.update(Pr, Gr, PV, DV)
        self.MP.update(Pr, PV, DV, TS)
        self.Tr.update(Pr, Gr, PV, DV)
        return

class HeldSuarezMoist(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.ForcingFactory(namelist)
        self.Sur = Surface.SurfaceFactory(namelist)
        self.MP  = Microphysics.MicrophysicsFactory(namelist)
        self.Co = Convection.ConvectionFactory(namelist)
        self.Tr = Turbulence.TurbulenceFactory(namelist)
        return


    def initialize(self, RS, Pr, Gr, PV, TS, namelist):
        Pr.QT_0      = namelist['forcing']['initial_surface_qt']
        Pr.T_0       = namelist['thermodynamics']['triple_point_temp']
        Pr.P_hw      = namelist['thermodynamics']['verical_half_width_of_the_q']
        Pr.phi_hw    = namelist['thermodynamics']['horizontal_half_width_of_the_q']

        Gamma        = namelist['forcing']['Gamma_init']
        Pr.sigma_b   = namelist['forcing']['sigma_b']
        Pr.k_a       = namelist['forcing']['k_a']
        Pr.k_b       = namelist['forcing']['k_b']
        Pr.k_s       = namelist['forcing']['k_s']
        Pr.k_f       = namelist['forcing']['k_f']
        Pr.DT_y      = namelist['forcing']['equator_to_pole_dT']
        Pr.T_equator = namelist['forcing']['equatorial_temperature']
        Pr.Dtheta_z  = namelist['forcing']['lapse_rate']
        Pr.T_pole    = namelist['forcing']['polar_temperature']
        Pr.init_k    = namelist['forcing']['initial_profile_power']
        eps_ = Pr.Rv/Pr.Rv-1.0

        PV.P.values = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),Pr.pressure_levels)
        if Pr.restart:
            RS.initialize(Pr, Gr, PV, TS, namelist)
        else:
            PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
            PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
            PV.P.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c'),Pr.pressure_levels)

            T_0 = 0.5 * (Pr.T_equator + Pr.T_pole) # eq. (18) Ullrich et al. (2014)
            B   = (T_0 - Pr.T_pole) / T_0 / Pr.T_pole # eq. (17) Ullrich et al. (2014)
            C   = 0.5 * (Pr.init_k + 2.0)*(Pr.T_equator-Pr.T_pole)/Pr.T_equator/Pr.T_pole # eq. (17) Ullrich et al. (2014)
            H = Pr.Rd*T_0/Pr.g

            for k in range(Pr.n_layers):
                p = 0.5*(Pr.pressure_levels[k]+Pr.pressure_levels[k+1])
                z = H*np.log(Pr.p_ref/p)
                Tau1 = 1.0/T_0*np.exp(Gamma*z/T_0) + B*(1-2.0*(z/2.0/H)**2.0) # eq. (14) Ullrich et al. (2014)
                Tau2 = C*(1.0-2.0*(z/2.0/H)**2.0)*np.exp(-(z/(2.0*H))**2.0) # eq. (15) Ullrich et al. (2014)
                for i in range(len(Gr.lat[:,0])):
                    I_T = (np.cos(Gr.lat[i,0])**Pr.init_k -
                              Pr.init_k/(Pr.init_k + 2.0)*np.cos(Gr.lat[i,0])**(Pr.init_k+2.0) )

                    Tv_ = 1.0/(Tau1 - Tau2 * I_T) # eq. (20) Ullrich et al. (2014)
                    for ii in range(Pr.nlons):
                        # eq. (A1) Tatcher and Jablonowski 2016
                        PV.QT.values[i,ii,k]  = (Pr.QT_0 * np.exp(-(Gr.lat[i,0] / Pr.phi_hw)**4.0)
                                  * np.exp(-((p/Pr.p_ref - 1.0) * Pr.p_ref / Pr.P_hw)**2.0))
                        PV.T.values[i,ii,k]  = Tv_/(1+eps_*PV.QT.values[i,ii,k]) # eq. (20) Ullrich et al. (2014)

        PV.physical_to_spectral(Pr, Gr)
        print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral[:,Pr.n_layers-1]).min())
        if namelist['initialize']['noise']:
             # calculate noise
             F0=np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
             fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0., noise_type=Pr.noise_type)
             noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(F0))*Pr.noise_amp
             # save noise here
             # np.save('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy',noise)
             # load noise here
             # noise = np.load('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy')
             # add noise
             PV.T.spectral[:,Pr.n_layers-1] = np.add(PV.T.spectral[:,Pr.n_layers-1],
                                                        Gr.SphericalGrid.grdtospec(noise))
        print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral[:,Pr.n_layers-1]).min())
        return

    def initialize_surface(self, Pr, Gr, PV, namelist):
        self.Sur.initialize(Pr, Gr, PV, namelist)
        return

    def initialize_convection(self, Pr, Gr, PV, namelist):
        self.Co.initialize(Pr, Gr, namelist)
        return

    def initialize_turbulence(self, Pr, namelist):
        self.Tr.initialize(Pr, namelist)
        return

    def initialize_forcing(self, Pr, Gr, namelist):
        self.Fo.initialize(Pr, Gr, namelist)
        return

    def initialize_microphysics(self, Pr, PV, DV,namelist):
        self.MP.initialize(Pr, PV, DV, namelist)
        return

    def initialize_io(self, Stats):
        CaseBase.initialize_io(self, Stats)
        self.Fo.initialize_io(Stats)
        self.Sur.initialize_io(Stats)
        self.MP.initialize_io(Stats)
        # self.Co.initialize_io(Stats)
        return

    def io(self, Pr, TS, Stats):
        CaseBase.io(self, Pr, TS, Stats)
        self.Fo.io(Pr, TS, Stats)
        self.Sur.io(Pr, TS, Stats)
        self.MP.io(Pr, TS, Stats)
        # self.Co.io(Pr, TS, Stats)
        return

    def stats_io(self, Gr, PV, Stats):
        CaseBase.stats_io(self, Gr, PV, Stats)
        self.Fo.stats_io(Stats)
        self.Sur.stats_io(Gr,Stats)
        self.MP.stats_io(Gr,PV, Stats)
        # self.Co.stats_io(Stats)
        return

    def update(self, Pr, Gr, PV, DV, TS):
        self.Sur.update(Pr, Gr, PV, DV)
        self.Fo.update(Pr, Gr, PV, DV)
        self.Tr.update(Pr, Gr, PV, DV)
        self.MP.update(Pr, PV, DV, TS)
        self.Co.update(Pr, Gr, PV, DV)
        return
