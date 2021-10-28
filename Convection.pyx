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
    if namelist['convection']['convection_model'] == 'None':
        return ConvectionNone(namelist)
    elif namelist['convection']['convection_model'] == 'RandomFlux':
        return ConvectionRandomFlux(namelist)
    elif namelist['convection']['convection_model'] == 'RandomTransport':
        return ConvectionRandomTransport(namelist)
    else:
        print('case not recognized')
    return

cdef class ConvectionBase:
    def __init__(self, namelist):
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return
    cpdef stats_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class ConvectionNone(ConvectionBase):
    def __init__(self, namelist):
        ConvectionBase.__init__(self, namelist)
        return
    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        return
    cpdef stats_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class ConvectionRandomFlux(ConvectionBase):
    def __init__(self, namelist):
        ConvectionBase.__init__(self, namelist)
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        cdef:
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm

        self.noise  = namelist['convection']['noise']
        Pr.Co_noise_magnitude  = namelist['convection']['noise_magnitude']
        Pr.Co_noise_correlation  = namelist['convection']['noise_correlation']
        Pr.Co_noise_type  = namelist['convection']['noise_type']
        Pr.Co_noise_lmin  = namelist['convection']['min_noise_wavenumber']
        Pr.Co_noise_lmax  = namelist['convection']['max_noise_wavenumber']
        Pr.Div_conv_amp  = namelist['convection']['Divergence_convective_noise_amplitude']
        Pr.Vort_conv_amp = namelist['convection']['Vorticity_convective_noise_amplitude']
        Pr.T_conv_amp    = namelist['convection']['T_convective_noise_amplitude']
        if Pr.moist_index > 0.0:
            Pr.QT_conv_amp   = namelist['convection']['QT_convective_noise_amplitude']

        self.F0 = np.zeros(nlm, dtype = np.complex, order='c')
        # fluxes are at pressure levels
        self.wT = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wVort = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wDiv = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        if Pr.moist_index > 0.0:
            self.wQT = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
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
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm
            double [:,:] dpi_grid    = np.zeros((nx, ny),  dtype = np.float64, order ='c')
            double complex [:,:] dpi = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
            double complex [:] Wp = np.zeros(nlm, dtype = np.complex, order ='c')

        if self.noise:
            # we compute the flux from below to the layer k skipping k=0
            for k in range(1,nl+1):
                dpi_grid = np.divide(1.0, np.subtract(PV.P.values[:,:,k],PV.P.values[:,:,k-1]))
                dpi.base[:,k] = Gr.SphericalGrid.grdtospec(dpi_grid.base)
                F0=np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
                sph_noise = spf.sphForcing(Pr.nlons,Pr.nlats, Pr.truncation_number,Pr.rsphere,
                                        Pr.Co_noise_lmin,Pr.Co_noise_lmax, Pr.Co_noise_magnitude,
                                        correlation = Pr.Co_noise_correlation, noise_type=Pr.Co_noise_type)
                Wp = sph_noise.forcingFn(F0)
                with nogil:
                    for i in range(nlm):
                        if k==nl:
                            self.wVort[i,k] = 0.0
                            self.wDiv[i,k]  = 0.0
                            # self.wT[i,k]    = Pr.Vort_conv_amp*Wp[i]*(PV.T.spectral[i,k-1] + PV.T.spectral[i,k-1])*0.5
                            if Pr.moist_index > 0.0:
                                self.wT[i,k]    = Pr.Vort_conv_amp*Wp[i]*(PV.QT.spectral[i,k-1] + PV.QT.spectral[i,k-1])*0.5
                        else:
                            self.wVort[i,k] = Pr.Vort_conv_amp*Wp[i]*(PV.Vorticity.spectral[i,k] - PV.Vorticity.spectral[i,k-1])*dpi[i,k]
                            self.wDiv[i,k]  = Pr.Vort_conv_amp*Wp[i]*(PV.Divergence.spectral[i,k] - PV.Divergence.spectral[i,k-1])*dpi[i,k]
                            # self.wT[i,k]    = Pr.Vort_conv_amp*Wp[i]*(PV.T.spectral[i,k] + PV.T.spectral[i,k-1])*0.5*dpi[i,k]
                            if Pr.moist_index > 0.0:
                                self.wT[i,k]    = Pr.Vort_conv_amp*Wp[i]*(PV.QT.spectral[i,k] + PV.QT.spectral[i,k-1])*0.5*dpi[i,k]

            # we compute the flux divergence at the middle of the k'th layer
            for k in range(nl):
                with nogil:
                    for i in range(nlm):
                        PV.Vorticity.ConvectiveFlux[i,k]  = -(self.wVort[i,k+1] - self.wVort[i,k])*dpi[i,k+1]
                        PV.Divergence.ConvectiveFlux[i,k] = -(self.wDiv[i,k+1]  - self.wDiv[i,k])*dpi[i,k+1]
                        # PV.T.ConvectiveFlux[i,k]          = -(self.wT[i,k+1]    - self.wT[i,k])*dpi[i,k+1]
                        if Pr.moist_index > 0.0:
                            PV.QT.ConvectiveFlux[i,k]     = -(self.wQT[i,k+1]   - self.wQT[i,k])*dpi[i,k+1]

        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
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


cdef class ConvectionRandomTransport(ConvectionBase):
    def __init__(self, namelist):
        ConvectionBase.__init__(self, namelist)
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        cdef:
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm

        self.noise  = namelist['convection']['noise']
        Pr.Co_noise_magnitude  = namelist['convection']['noise_magnitude']
        Pr.Co_noise_correlation  = namelist['convection']['noise_correlation']
        Pr.Co_noise_type  = namelist['convection']['noise_type']
        Pr.Co_noise_lmin  = namelist['convection']['min_noise_wavenumber']
        Pr.Co_noise_lmax  = namelist['convection']['max_noise_wavenumber']
        Pr.Div_conv_amp  = namelist['convection']['Divergence_convective_noise_amplitude']
        Pr.Vort_conv_amp = namelist['convection']['Vorticity_convective_noise_amplitude']
        Pr.T_conv_amp    = namelist['convection']['T_convective_noise_amplitude']
        if Pr.moist_index > 0.0:
            Pr.QT_conv_amp   = namelist['convection']['QT_convective_noise_amplitude']

        self.F0 = np.zeros(nlm, dtype = np.complex, order='c')
        # fluxes are at pressure levels
        Wp = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wT = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wVort = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        self.wDiv = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
        if Pr.moist_index > 0.0:
            self.wQT = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
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
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm
            double [:,:] dpi_grid    = np.zeros((nx, ny),  dtype = np.float64, order ='c')
            double complex [:,:] dpi = np.zeros((nlm, nl+1), dtype = np.complex, order ='c')
            double complex [:] Wp = np.zeros(nlm, dtype = np.complex, order ='c')

        if self.noise:
            #print('calculate convective fluxes here')
            # we compute the flux from below to the layer k skipping k=0
            for k in range(1,nl+1):
                dpi_grid = np.divide(1.0, np.subtract(PV.P.values[:,:,k],PV.P.values[:,:,k-1]))
                dpi.base[:,k] = Gr.SphericalGrid.grdtospec(dpi_grid.base)
                F0=np.zeros(Gr.SphericalGrid.nlm,dtype = np.complex, order='c')
                sph_noise = spf.sphForcing(Pr.nlons,Pr.nlats, Pr.truncation_number,Pr.rsphere,
                                        Pr.Co_noise_lmin,Pr.Co_noise_lmax, Pr.Co_noise_magnitude,
                                        correlation = Pr.Co_noise_correlation, noise_type=Pr.Co_noise_type)
                Wp = sph_noise.forcingFn(F0)
                with nogil:
                    for i in range(nlm):
                        if k==nl:
                            self.wVort[i,k] = 0.0
                            self.wDiv[i,k]  = 0.0
                            self.wT[i,k]    = Pr.T_conv_amp*Wp[i]*dpi[i,k]
                            if Pr.moist_index > 0.0:
                                self.wQT[i,k] = Pr.QT_conv_amp*Wp[i]*dpi[i,k]
                        else:
                            self.wVort[i,k] = Pr.Vort_conv_amp*Wp[i]*dpi[i,k]
                            self.wDiv[i,k]  = Pr.Div_conv_amp*Wp[i]*dpi[i,k]
                            self.wT[i,k]    = Pr.T_conv_amp*Wp[i]*dpi[i,k]
                            if Pr.moist_index > 0.0:
                                self.wQT[i,k] = Pr.QT_conv_amp*Wp[i]*dpi[i,k]
            # we compute the flux divergence at the middle of the k'th layer
            for k in range(nl):
                with nogil:
                    for i in range(nlm):
                        PV.Vorticity.ConvectiveFlux[i,k]  = -(self.wVort[i,k+1] - self.wVort[i,k])*dpi[i,k+1]
                        PV.Divergence.ConvectiveFlux[i,k] = -(self.wDiv[i,k+1]  - self.wDiv[i,k])*dpi[i,k+1]
                        # PV.T.ConvectiveFlux[i,k]          = -(self.wT[i,k+1]    - self.wT[i,k])*dpi[i,k+1]
                        if Pr.moist_index > 0.0:
                            PV.QT.ConvectiveFlux[i,k]     = -(self.wQT[i,k+1]   - self.wQT[i,k])*dpi[i,k+1]

        return

    cpdef stats_io(self, NetCDFIO_Stats Stats):
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
