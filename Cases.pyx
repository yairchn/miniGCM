import numpy as np
from PrognosticVariables cimport PrognosticVariables
from PrognosticVariables cimport PrognosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
cimport Forcing
cimport Surface
cimport Microphysics
import cython
import sys
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

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

    cpdef initialize_forcing(self, Parameters Pr, namelist):
        return

    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, namelist):
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        return

    cpdef update_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return

    cpdef update_forcing(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return

    cpdef update_microphysics(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TimeStepping TS):
        return

cdef class HeldSuarez(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.HelzSuarez()
        self.Sur = Surface.SurfaceNone()
        self.MP = Microphysics.MicrophysicsNone()
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        PV.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        PV.T_init        = np.array([229.0, 257.0, 295.0])

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
        PV.physical_to_spectral(Pr, Gr)
        # # initilize spectral values
        # for k in range(Pr.n_layers):
        #     PV.P.spectral.base[:,k]           = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,k])
        #     PV.T.spectral.base[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values[:,:,k])
        #     PV.QT.spectral.base[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values[:,:,k])
        #     PV.Vorticity.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values[:,:,k])
        #     PV.Divergence.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values[:,:,k])
        # PV.P.spectral[:,Pr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values[:,:,Pr.n_layers])
        return

    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        self.Sur.initialize(Pr, Gr, PV, namelist)
        return

    cpdef initialize_forcing(self, Parameters Pr, namelist):
        self.Fo.initialize(Pr, namelist)
        return

    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, namelist):
        self.MP.initialize(Pr, PV, namelist)
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

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        CaseBase.stats_io(self, Stats)
        self.Fo.stats_io(Stats)
        self.Sur.stats_io(Stats)
        return

    cpdef update_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        self.Sur.update(Pr, Gr, PV, DV)
        return

    cpdef update_forcing(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        self.Fo.update(Pr, Gr, PV, DV)
        return

    cpdef update_microphysics(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TimeStepping TS):
        self.MP.update(Pr, TS, PV)
        return

cdef class HeldSuarezMoist(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.HelzSuarezMoist()
        self.Sur = Surface.SurfaceBulkFormula()
        self.MP = Microphysics.MicrophysicsCutoff()
        return


    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        cdef:
            Py_ssize_t i, j, k
            double Gamma, T_0, B, C, H
            double [:] z
            double [:] tau_z_1
            double [:] tau_z_2
            double [:] tau_z_3
            double [:] tau_1
            double [:] tau_2
            double [:] tau_int_1
            double [:] tau_int_2
            double [:] I_T
            double [:] QT_meridional
            double [:] T_meridional
            double [:,:] p
            double [:,:,:] qv_star_
            double [:,:,:] Tv_
            double [:,:,:] QT_


        Pr.QT_0      = namelist['forcing']['initial_surface_qt']
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

        PV.P_init            = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        PV.T_init            = np.array([245.0, 282.0, 303.0])
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

        H = Pr.Rd*T_0/Pr.g
        tau_z_1   = np.exp(np.divide(np.multiply(Gamma,z), T_0))
        tau_z_2   = 1.0 - np.multiply(2.0, np.power(np.divide(z,2.0*H),2.0))
        tau_z_3   = np.exp(-np.power(np.divide(z, 2.0*H),2.0))
        tau_1     = np.add(np.multiply(np.divide(1.0 , T_0) , tau_z_1), np.multiply(np.multiply(B , tau_z_2) , tau_z_3))
        tau_2     = np.multiply(np.multiply(C , tau_z_2) , tau_z_3)
        tau_int_1 = np.add(np.multiply(1.0 / Gamma , np.subtract(tau_z_1, 1.0)), np.multiply(np.multiply(B , z ), tau_z_3))
        tau_int_2 = np.multiply(np.multiply(C , z) ,tau_z_3)
        I_T       = np.subtract(np.power(np.cos(Gr.lat[:,0]),Pr.init_k),
                    np.multiply(Pr.init_k/(Pr.init_k + 2.0), np.power(np.cos(Gr.lat[:,0]),Pr.init_k+2.0)))

        for i in range(len(Gr.lat[:,0])):
            for j in range(len(z)):
                p[i,j] = Pr.p_ref * np.exp(-Pr.g / Pr.Rd * (tau_int_1[j] - tau_int_2[j] * I_T[i]))
                for k in range(Pr.n_layers):
                    if (p[i,j]>PV.P_init[k] and p[i,j]<PV.P_init[k+1]):
                        Tv_[i,j,k] = np.power(tau_1[j] - tau_2[j] * I_T[i],-1.0)
                        QT_[i,j,k] = (Pr.QT_0 * np.exp(-(Gr.lat[i,0] / Pr.phi_hw)**4.0)
                                    * np.exp(-((p[i,j]/Pr.p_ref - 1.0) * Pr.p_ref / Pr.P_hw)**2.0))
                    else:
                        Tv_[i,j,k] = np.nan
                        QT_[i,j,k] = np.nan

        for k in range(Pr.n_layers):
            QT_meridional = (Pr.QT_0 * np.exp(-np.power(np.divide(Gr.lat[:,0] , Pr.phi_hw),4.0))
                * np.exp(-((PV.P_init[k]/Pr.p_ref - 1.0) * Pr.p_ref / Pr.P_hw)**2.0))
            # QT_meridional = np.nanmean(QT_[:,:,k], axis=1)
            eps = 1.0/Pr.eps_v-1.0
            T_meridional = np.nanmean(np.divide(Tv_[:,:,k],np.add(1,np.multiply(0.608,QT_[:,:,k]))), axis=1)
            for i in range(Pr.nlons):
                PV.QT.values.base[:,i,k] = QT_meridional
                PV.T.values.base[:,i,k]  = T_meridional

        for k in range(Pr.n_layers):
            PV.T.spectral.base[:,k]           = Gr.SphericalGrid.grdtospec(PV.T.values.base[:,:,k])
            PV.QT.spectral.base[:,k]          = Gr.SphericalGrid.grdtospec(PV.QT.values.base[:,:,k])
            PV.Vorticity.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(PV.Vorticity.values.base[:,:,k])
            PV.Divergence.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(PV.Divergence.values.base[:,:,k])
        PV.P.spectral.base[:,Pr.n_layers]     = Gr.SphericalGrid.grdtospec(PV.P.values.base[:,:,Pr.n_layers])
        return

    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        self.Sur.initialize(Pr, Gr, PV, namelist)
        return

    cpdef initialize_forcing(self, Parameters Pr, namelist):
        self.Fo.initialize(Pr, namelist)
        return

    cpdef initialize_microphysics(self, Parameters Pr, PrognosticVariables PV, namelist):
        self.MP.initialize(Pr, PV, namelist)
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
        self.MP.io(Pr, TS, Stats)
        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
        CaseBase.stats_io(self, Stats)
        self.Fo.stats_io(Stats)
        self.Sur.stats_io(Stats)
        self.MP.stats_io(Stats)
        return

    cpdef update_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        self.Sur.update(Pr, Gr, PV, DV)
        return

    cpdef update_forcing(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        self.Fo.update(Pr, Gr, PV, DV)
        return

    cpdef update_microphysics(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TimeStepping TS):
        self.MP.update(Pr, TS, PV)
        return
