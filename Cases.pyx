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
    if namelist['meta']['casename'] == 'Default':
        return Default(namelist)
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

    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
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

cdef class Default(CaseBase):
    def __init__(self, namelist):
        # Pr.casename = namelist['meta']['casename']
        self.Fo  = Forcing.Default()
        self.Sur = Surface.SurfaceBulkFormula()
        self.MP = Microphysics.MicrophysicsNone()
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, namelist):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            double [:,:] noise
            double [:,:] fr

        Pr.k_a = namelist['forcing']['k_a']
        Pr.k_b = namelist['forcing']['k_b']
        Pr.k_s = namelist['forcing']['k_s']
        Pr.k_f = namelist['forcing']['k_f']
        Pr.DH_y = namelist['forcing']['equator_to_pole_dH']
        Pr.Dtheta_z = namelist['forcing']['lapse_rate']
        Pr.H_equator = namelist['forcing']['equatorial_depth']

        PV.Vorticity.values  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.Divergence.values = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.QT.values         = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),Pr.QT_init)
        PV.H.values          = np.multiply(np.ones((Pr.nlats, Pr.nlons, Pr.n_layers),   dtype=np.double, order='c'),Pr.H_init)
        for i  in range(nx):
            for j  in range(ny):
                PV.H.values[i,j,2] -= 1.0*np.exp(-((Gr.lat[i,j] - Gr.lat[-1,j])**2.0)/(2.0*0.3)**2.0)
                PV.H.values[i,j,1] += 1.0*np.exp(-((Gr.lat[i,j] - Gr.lat[-1,j])**2.0)/(2.0*0.3)**2.0)
        PV.physical_to_spectral(Pr, Gr)
        print('layer 3 Depth min',Gr.SphericalGrid.spectogrd(PV.H.spectral.base[:,Pr.n_layers-1]).min())
        if Pr.inoise==1:
             # calculate noise
             F0=np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
             H_noise = spf.sphForcing(Pr.nlons,Pr.nlats,Pr.truncation_number,Pr.rsphere,lmin= 1, lmax= 100, magnitude = 0.05, correlation = 0., noise_type=Pr.noise_type)
             noise = Gr.SphericalGrid.spectogrd(H_noise.forcingFn(F0))*Pr.noise_amp
             # save noise here
             # np.save('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy',noise)
             # load noise here
             # noise = np.load('./norm_rand_grid_noise_'+Pr.noise_type+'_.npy')
             # add noise
             PV.H.spectral.base[:,Pr.n_layers-1] = np.add(PV.H.spectral.base[:,Pr.n_layers-1],
                                                        Gr.SphericalGrid.grdtospec(noise.base))
        print('layer 3 Depth min',Gr.SphericalGrid.spectogrd(PV.H.spectral.base[:,Pr.n_layers-1]).min())
        return

    cpdef initialize_surface(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        self.Sur.initialize(Pr, Gr, PV, DV, namelist)
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