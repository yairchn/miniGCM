#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
import sphericalForcing as spf

def ConvectionFactory(namelist):
    if namelist['convection']['convection_type'] == 'None':
        return ConvectionNone(namelist)
    elif namelist['convection']['convection_type'] == 'Randon':
        return ConvectionRandomNoise(namelist)
    else:
        print('case not recognized')
    return

cdef class ConvectionBase:
    def __init__(self, namelist):
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class ConvectionNone(ConvectionBase):
    def __init__(self):
        ConvectionBase.__init__(self)
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class ConvectionRandomNoise(ConvectionBase):
    def __init__(self):
        ConvectionBase.__init__(self)
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        cdef:
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm

        Pr.T_conv_amp    = namelist['convection']['convective_noise_amplitude']
        Pr.QT_conv_amp   = namelist['convection']['convective_noise_amplitude']
        Pr.Vort_conv_amp = namelist['convection']['convective_noise_amplitude']
        Pr.Div_conv_amp  = namelist['convection']['convective_noise_amplitude']
        self.F0          = np.zeros(nlm, dtype = np.complex, order='c')
        # fluxes are at pressure levels
        self.wT = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wQT = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wVort = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wDiv = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_conv_wT')
        Stats.add_global_mean('global_mean_conv_wQT')
        Stats.add_global_mean('global_mean_conv_wVort')
        Stats.add_global_mean('global_mean_conv_wDiv')
        Stats.add_zonal_mean('zonal_mean_conv_wT')
        Stats.add_zonal_mean('zonal_mean_conv_wQT')
        Stats.add_zonal_mean('zonal_mean_conv_wVort')
        Stats.add_zonal_mean('zonal_mean_conv_wDiv')
        Stats.add_meridional_mean('meridional_mean_conv_wT')
        Stats.add_meridional_mean('meridional_mean_conv_wQT')
        Stats.add_meridional_mean('meridional_mean_conv_wVort')
        Stats.add_meridional_mean('meridional_mean_conv_wDiv')
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm
            double [:,:] dpi_grid    = np.zeros((nx, ny),  dtype = np.float64, order ='c')
            double complex [:,:] dpi = np.zeros((nlm, nl), dtype = np.complex, order ='c')

        for k in range(nl):
            dpi_grid = np.divide(np.subtract(PV.P.value[:,:,k+1],PV.P.value[:,:,k]))
            dpi[:,k] = Gr.SphericalGrid.grdtospec(dpi_grid.base)
            self.sph_noise = spf.sphForcing(nx, ny, Pr.truncation_number, Pr.rsphere,lmin= 1,
                                            lmax= 100, magnitude = 0.05, correlation = 0., noise_type='local')
            self.Wp[:,k] = self.sph_noise.forcingFn(self.F0)*Pr.conv_amp
            with nogil:
                for i in range(nlm):
                    if k==nl-1:
                        self.wT[i,k+1]    = Pr.T_conv_amp*self.Wp[i,k+1]*(PV.T.spectral[i,k]   + PV.T.spectral[i,k])*0.5
                        self.wQT[i,k+1]   = Pr.QT_conv_amp*self.Wp[i,k+1]*(PV.QT.spectral[i,k] + PV.QT.spectral[i,k])*0.5
                        self.wVort[i,k+1] = 0.0
                        self.wDiv[i,k+1]  = 0.0
                    else:
                        self.wT[i,k+1]    = Pr.T_conv_amp*self.Wp[i,k+1]*(PV.T.spectral[i,k+1]   + PV.T.spectral[i,k])*0.5
                        self.wQT[i,k+1]   = Pr.QT_conv_amp*self.Wp[i,k+1]*(PV.QT.spectral[i,k+1] + PV.QT.spectral[i,k])*0.5
                        self.wVort[i,k+1] = Pr.Vort_conv_amp*self.Wp[i,k+1]*(PV.Vorticity.spectral[i,k+1] - PV.Vorticity.spectral[i,k])
                        self.wDiv[i,k+1]  = Pr.Div_conv_amp*self.Wp[i,k+1]*(PV.Divergence.spectral[i,k+1] - PV.Divergence.spectral[i,k])
        with nogil:
            for k in range(nl):
                for i in range(nlm):
                    PV.T.ConvectiveFlux[i,k]          = -(self.wT[i,k+1]    - self.wT[i,k])*dpi[i,k]
                    PV.QT.ConvectiveFlux[i,k]         = -(self.wQT[i,k+1]   - self.wQT[i,k])*dpi[i,k]
                    PV.Vorticity.ConvectiveFlux[i,k]  = -(self.wVort[i,k+1] - self.wVort[i,k])*dpi[i,k]
                    PV.Divergence.ConvectiveFlux[i,k] = -(self.wDiv[i,k+1]  - self.wDiv[i,k])*dpi[i,k]

        return

    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        Stats.write_zonal_mean('zonal_mean_conv_wT',self.wT[:,1:])
        Stats.write_zonal_mean('zonal_mean_conv_wQT',self.wQT[:,1:])
        Stats.write_zonal_mean('zonal_mean_conv_wVort',self.wVort[:,1:])
        Stats.write_zonal_mean('zonal_mean_conv_wDiv',self.wDiv[:,1:])
        Stats.write_meridional_mean('meridional_mean_conv_wT',self.wT[:,1:])
        Stats.write_meridional_mean('meridional_mean_conv_wQT',self.wQT[:,1:])
        Stats.write_meridional_mean('meridional_mean_conv_wVort',self.wVort[:,1:])
        Stats.write_meridional_mean('meridional_mean_conv_wDiv',self.wDiv[:,1:])
        Stats.write_surface_zonal_mean('zonal_mean_conv_wT',self.wT[:,1:])
        Stats.write_surface_zonal_mean('zonal_mean_conv_wQT',self.wQT[:,1:])
        Stats.write_surface_zonal_mean('zonal_mean_conv_wVort',self.wVort[:,1:])
        Stats.write_surface_zonal_mean('zonal_mean_conv_wDiv',self.wDiv[:,1:])
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t nl = Pr.n_layers
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'conv_wT',self.wT[:,1:])
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'conv_wQT',self.wQT[:,1:])
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'conv_wVort',self.wVort[:,1:])
        Stats.write_3D_variable(Pr, int(TS.t),nl, 'conv_wDiv',self.wDiv[:,1:])
        return
