#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
import sys
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

cdef extern from "microphysics_functions.h":
    void microphysics_cutoff(double cp, double dt, double Rv, double Lv, double T_0, double rho_w,
           double g, double max_ss, double pv_star0, double eps_v, double* p, double* T,
           double* qt, double* ql, double* T_mp, double* qt_mp, double* rain_rate, double* qv_star,
           Py_ssize_t imax, Py_ssize_t jmax, Py_ssize_t kmax) nogil

def MicrophysicsFactory(namelist):
    if namelist['microphysics']['microphysics_model'] == 'None':
        return MicrophysicsNone(namelist)
    elif namelist['microphysics']['microphysics_model'] == 'Cutoff':
        if namelist['thermodynamics']['thermodynamics_type'] == 'dry':
             sys.exit('Cannot run microphysics with dry thermodynamics')
        return MicrophysicsCutoff(namelist)
    else:
        print('case not recognized')
    return

cdef class MicrophysicsBase:
    def __init__(self, namelist):
        return
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, Grid Gr, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class MicrophysicsNone(MicrophysicsBase):
    def __init__(self, namelist):
        MicrophysicsBase.__init__(self, namelist)
        return
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        PV.QT.mp_tendency = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        PV.T.mp_tendency  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        return
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, Grid Gr, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class MicrophysicsCutoff(MicrophysicsBase):
    def __init__(self, namelist):
        MicrophysicsBase.__init__(self, namelist)
        return

    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        cdef:
            Py_ssize_t k
            Py_ssize_t nl = Pr.n_layers
            double [:,:] P_half
        self.RainRate = np.zeros((Pr.nlats, Pr.nlons),  dtype=np.double, order='c')
        self.qv_star  = np.zeros((Pr.nlats, Pr.nlons, Pr.n_layers),  dtype=np.double, order='c')
        Pr.max_ss    =  namelist['microphysics']['max_supersaturation']
        Pr.rho_w     =  namelist['microphysics']['water_density']
        Pr.mp_dt     =  namelist['microphysics']['autoconversion_timescale']
        for k in range(nl):
            P_half = np.multiply(np.add(PV.P.values[:,:,k],PV.P.values[:,:,k+1]),0.5)
            # Clausius–Clapeyron equation based saturation
            qv_star = np.multiply(np.divide(np.multiply(Pr.pv_star0,Pr.eps_v),P_half),
                np.exp(-np.multiply(np.divide(Pr.Lv,Pr.Rv),np.subtract(np.divide(1.0,PV.T.values[:,:,k]),np.divide(1.0,Pr.T_0)))))

            DV.QL.values.base[:,:,k] = np.clip(PV.QT.values[:,:,k] - qv_star,0.0, None)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_dQTdt')
        Stats.add_surface_global_mean('global_mean_RainRate')
        Stats.add_zonal_mean('zonal_mean_dQTdt')
        Stats.add_meridional_mean('meridional_mean_dQTdt')
        Stats.add_surface_zonal_mean('zonal_mean_RainRate')
        Stats.add_global_mean('global_mean_qv_star')
        Stats.add_zonal_mean('zonal_mean_qv_star')
        Stats.add_meridional_mean('meridional_mean_qv_star')
        return

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        cdef:
            Py_ssize_t nx = Pr.nlats
            Py_ssize_t ny = Pr.nlons
            Py_ssize_t nl = Pr.n_layers

        if (TS.t%Pr.mp_dt == 0.0):
            with nogil:
                microphysics_cutoff(Pr.cp, Pr.mp_dt, Pr.Rv, Pr.Lv, Pr.T_0, Pr.rho_w,
                                    Pr.g, Pr.max_ss, Pr.pv_star0, Pr.eps_v, &PV.P.values[0,0,0],
                                    &PV.T.values[0,0,0], &PV.QT.values[0,0,0], &DV.QL.values[0,0,0], &PV.T.mp_tendency[0,0,0],
                                    &PV.QT.mp_tendency[0,0,0], &self.RainRate[0,0], &self.qv_star[0,0,0],
                                    nx, ny, nl)
        else:
            PV.QT.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            PV.T.mp_tendency = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
            self.RainRate = np.zeros((nx,ny),dtype=np.float64, order='c')
            self.qv_star = np.zeros((nx,ny,nl),dtype=np.float64, order='c')


        return

    cpdef stats_io(self, Grid Gr, PrognosticVariables PV, NetCDFIO_Stats Stats):
        Stats.write_global_mean(Gr, 'global_mean_dQTdt', PV.QT.mp_tendency)
        Stats.write_surface_global_mean(Gr, 'global_mean_RainRate', self.RainRate)
        Stats.write_zonal_mean('zonal_mean_dQTdt',PV.QT.mp_tendency)
        Stats.write_meridional_mean('meridional_mean_dQTdt',PV.QT.mp_tendency)
        Stats.write_surface_zonal_mean('zonal_mean_RainRate',self.RainRate)
        Stats.write_global_mean(Gr, 'global_mean_qv_star',self.qv_star)
        Stats.write_zonal_mean('zonal_mean_qv_star',self.qv_star)
        Stats.write_meridional_mean('meridional_mean_qv_star',self.qv_star)
        return

    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_2D_variable(Pr, int(TS.t) , 'Rain_Rate',self.RainRate)
        return
