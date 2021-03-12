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

def CasesFactory(namelist):
    if namelist['meta']['casename'] == 'DryVortex':
        return DryVortex(namelist)
    elif namelist['meta']['casename'] == 'MoistVortex':
        return MoistVortex(namelist)
    elif namelist['meta']['casename'] == 'ReedJablonowski':
        return ReedJablonowski(namelist)
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

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return

    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return

cdef class DryVortex(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.ForcingBettsMiller()
        self.Sur = Surface.SurfaceNone()
        self.MP = Microphysics.MicrophysicsNone()
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        cdef:
            Py_ssize_t nx = Pr.nx
            Py_ssize_t ny = Pr.ny
            Py_ssize_t nl = Pr.n_layers
            double x0 = Gr.nx/2.0
            double y0 = Gr.ny/2.0
            double [:,:] noise

        PV.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        PV.T_init        = np.array([229.0, 259.0, 291.0])
        PV.QT_init        = np.array([2.5000e-04, 0.0016, 0.0115])
        Pr.amp_dTdp        = np.array([0.2, 1.0, 0.0])

        Pr.sigma_T = namelist['initialize']['warm_core_width']
        Pr.amp_T = namelist['initialize']['warm_core_amplitude']

        PV.U.values  = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        PV.V.values  = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        PV.QT.values = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl),   dtype=np.double, order='c'),PV.QT_init)
        PV.P.values  = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl+1), dtype=np.double, order='c'),PV.P_init)
        PV.T.values  = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl),   dtype=np.double, order='c'),PV.T_init)

        for i in range(nx):
            for j in range(ny):
                for k in range(nl):
                    PV.T.values[i,j,k] += Pr.amp_T*Pr.amp_dTdp[k]*np.exp(-((Gr.x[i] - Gr.x[Gr.xc])**2.0+(Gr.y[j] - Gr.y[Gr.yc])**2.0)/(2.0*Pr.amp_T**2.0))


        # if Pr.inoise==1: # load the random noise to grid space
        #      noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')/10.0
        #      PV.T.values.base[:,:,Pr.n_layers-1] = np.add(PV.T.values.base[:,:,Pr.n_layers-1],noise.base)
        # print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
        # if Pr.inoise==1:
        #      # load the random noise to grid space
        #      #noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')*Pr.noise_amp
        #      # calculate noise
        #      spec_zeros = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
        #      fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0.)
        #      noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(spec_zeros))*Pr.noise_amp
        #      # add noise
        #      PV.T.spectral.base[:,Pr.n_layers-1] = np.add(PV.T.spectral.base[:,Pr.n_layers-1],
        #                                                 Gr.SphericalGrid.grdtospec(noise.base))
        # print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
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
            Py_ssize_t nx = Gr.nx
            Py_ssize_t ny = Gr.ny
            Py_ssize_t nl = Gr.nl
            double x0 = Gr.nx/2.0 + Gr.ng
            double y0 = Gr.ny/2.0 + Gr.ng
            double [:,:] noise

        PV.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        PV.T_init        = np.array([229.0, 259.0, 291.0])
        PV.QT_init        = np.array([2.5000e-04, 0.0016, 0.0115])
        PV.amp_dTdp        = np.array([0.2, 1.0, 0.0])

        PV.U.values  = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        PV.V.values = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        PV.QT.values         = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl),   dtype=np.double, order='c'),PV.QT_init)
        PV.P.values          = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl+1), dtype=np.double, order='c'),PV.P_init)
        PV.T.values          = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl),   dtype=np.double, order='c'),PV.T_init)

        for i in range(nx):
            for j in range(ny):
                for k in range(nl):
                    PV.T.values[i,j,k] += (Pr.amp_T*PV.amp_dTdp[k]
                                          *np.exp(-((Gr.x[i] - x0)**2.0+(Gr.y[j] - y0)**2.0)/
                                            (2.0*Pr.amp_T**2.0)))


        # if Pr.inoise==1: # load the random noise to grid space
        #      noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')/10.0
        #      PV.T.values.base[:,:,Pr.n_layers-1] = np.add(PV.T.values.base[:,:,Pr.n_layers-1],noise.base)
        # print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
        # if Pr.inoise==1:
        #      # load the random noise to grid space
        #      #noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')*Pr.noise_amp
        #      # calculate noise
        #      spec_zeros = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
        #      fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0.)
        #      noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(spec_zeros))*Pr.noise_amp
        #      # add noise
        #      PV.T.spectral.base[:,Pr.n_layers-1] = np.add(PV.T.spectral.base[:,Pr.n_layers-1],
        #                                                 Gr.SphericalGrid.grdtospec(noise.base))
        # print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
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

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        CaseBase.io(self, Pr, Gr, TS, Stats)
        self.Fo.io(Pr, Gr, TS, Stats)
        self.Sur.io(Pr, Gr, TS, Stats)
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

cdef class ReedJablonowski(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.ForcingBettsMiller()
        self.Sur = Surface.SurfaceNone()
        self.MP = Microphysics.MicrophysicsNone()
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        cdef:
            double [:,:] noise

        # PV.P_init        = np.array([Pr.p1, Pr.p2, Pr.p3, Pr.p_ref])
        # PV.T_init        = np.array([229.0, 259.0, 291.0])
        # PV.QT_init        = np.array([2.5000e-04, 0.0016, 0.0115])

        # Pr.sigma_T = namelist['forcing']['sigma_T']
        # Pr.amp_T = namelist['forcing']['amp_T']

        # PV.Vorticity.values  = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        # PV.Divergence.values = np.zeros((Gr.nx, Gr.ny, Gr.nl),  dtype=np.double, order='c')
        # PV.QT.values         = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl),   dtype=np.double, order='c'),PV.QT_init)
        # PV.P.values          = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl+1), dtype=np.double, order='c'),PV.P_init)
        # PV.T.values          = np.multiply(np.ones((Gr.nx, Gr.ny, Gr.nl),   dtype=np.double, order='c'),PV.T_init)
        # for i in range(nx):
        #     for j in range(ny):
        #         for k in range(nl):

        #             qt[i,j,k] = qt_0*exp(-z[k]/z_qt1)*exp(-(z[k]/z_qt2)**2.0)
        #             Tv[i,j,k] = Tv_0 - Gamma*z[k]
        #             T[i,j,k] = Tv[i,j,k]/(1.0+0.608*qt[i,j,k])
        #             p[i,j,k] = ((Tv_0-Gamma*z[k])/Tv_0)**(g/(Rd*Gamma))


        # if Pr.inoise==1: # load the random noise to grid space
        #      noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')/10.0
        #      PV.T.values.base[:,:,Pr.n_layers-1] = np.add(PV.T.values.base[:,:,Pr.n_layers-1],noise.base)
        # print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
        # if Pr.inoise==1:
        #      # load the random noise to grid space
        #      #noise = np.load('./Initial_conditions/norm_rand_grid_noise_white.npy')*Pr.noise_amp
        #      # calculate noise
        #      spec_zeros = np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
        #      fr = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0.)
        #      noise = Gr.SphericalGrid.spectogrd(fr.forcingFn(spec_zeros))*Pr.noise_amp
        #      # add noise
        #      PV.T.spectral.base[:,Pr.n_layers-1] = np.add(PV.T.spectral.base[:,Pr.n_layers-1],
        #                                                 Gr.SphericalGrid.grdtospec(noise.base))
        # print('layer 3 Temperature min',Gr.SphericalGrid.spectogrd(PV.T.spectral.base[:,Pr.n_layers-1]).min())
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

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        CaseBase.io(self, Pr, Gr, TS, Stats)
        self.Fo.io(Pr, Gr, TS, Stats)
        self.Sur.io(Pr, Gr, TS, Stats)
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