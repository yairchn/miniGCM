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
from scipy.interpolate import pchip_interpolate
from libc.math cimport pow, cbrt, exp, fmin, fmax

# include 'parameters.pxi'


# INTERP
# cpdair - cp of dry air

# Note: the RRTM modules are compiled in the 'RRTMG' directory:
# cdef extern:
#     void c_rrtmg_lw_init(double *cpdair)
#     void c_mcica_subcol_lw (int *iplon, int *ncol, int *nlay, int *icld, int *permuteseed, int *irng, 
#              double *play, double *cldfrac, double *ciwp, double *clwp, double *rei, double *rel, double *tauc, 
#              double *cldfmcl, double *ciwpmcl, double *clwpmcl, double *reicmcl, double *relqmcl, double *taucmcl)
#     void c_rrtmg_lw (int *ncol    ,int *nlay    ,int *icld    ,int *idrv    ,
#              double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
#              double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
#              double *cfc11vmr,double *cfc12vmr,double *cfc22vmr,double *ccl4vmr ,double *emis    ,
#              int *inflglw ,int *iceflglw,int *liqflglw,double *cldfr   ,
#              double *taucld  ,double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
#              double *tauaer  ,
#              double *uflx    ,double *dflx    ,double *hr      ,double *uflxc   ,double *dflxc,  double *hrc,
#              double *duflx_dt,double *duflxc_dt)
#     void c_rrtmg_sw_init(double *cpdair)
#     void c_mcica_subcol_sw(int *iplon, int *ncol, int *nlay, int *icld, int *permuteseed, int *irng, 
#              double *play, double *cldfrac, double *ciwp, double *clwp, double *rei, double *rel, 
#              double *tauc, double *ssac, double *asmc, double *fsfc,
#              double *cldfmcl, double *ciwpmcl, double *clwpmcl, double *reicmcl, double *relqmcl,
#              double *taucmcl, double *ssacmcl, double *asmcmcl, double *fsfcmcl)
#     void c_rrtmg_sw (int *ncol    ,int *nlay    ,int *icld    ,int *iaer    ,
#              double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
#              double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
#              double *asdir   ,double *asdif   ,double *aldir   ,double *aldif   ,
#              double *coszen  ,double *adjes   ,int *dyofyr  ,double *scon    ,
#              int *inflgsw ,int *iceflgsw,int *liqflgsw,double *cldfr   ,
#              double *taucld  ,double *ssacld  ,double *asmcld  ,double *fsfcld  ,
#              double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
#              double *tauaer  ,double *ssaaer  ,double *asmaer  ,double *ecaer   ,
#              double *swuflx  ,double *swdflx  ,double *swhr    ,double *swuflxc ,double *swdflxc ,double *swhrc)


def RadiationFactory(namelist):
    if namelist['radiation']['radiation_model'] == 'None':
        return RadiationNone(namelist)
    # elif namelist['radiation']['radiation_model'] == 'RRTMG':
    #     return RadiationRRTMG(namelist)
    else:
        print('case not recognized')
    return

cdef class RadiationBase:
    def __init__(self, namelist):
        return
    cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        return
    cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

cdef class RadiationNone(RadiationBase):
    def __init__(self, namelist):
        RadiationBase.__init__(self, namelist)
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
    cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
        return
    cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
        return

# cdef class RadiationRRTMG(RadiationBase):
#     def __init__(self, namelist):
#         RadiationBase.__init__(self, namelist)
#         return

#     cpdef initialize(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
#         cdef:
#             Py_ssize_t k
#             Py_ssize_t nl = Pr.n_layers
#             double [:,:] P_half
#             double cpdair = np.float64(Pr.cp)

#         self.srf_lw_down = 0.0
#         self.srf_sw_down = 0.0
#         self.srf_lw_up = 0.0
#         self.srf_sw_down = 0.0
#         c_rrtmg_lw_init(&cpdair)
#         c_rrtmg_sw_init(&cpdair)

#         try:
#             self.patch_pressure = namelist['radiation']['patch_pressure']
#         except:
#             self.patch_pressure = 1000.00*100.0

#         # Namelist options related to gas concentrations
#         try:
#             self.co2_factor = namelist['radiation']['co2_factor']
#         except:
#             self.co2_factor = 1.0
#         try:
#             self.h2o_factor = namelist['radiation']['h2o_factor']
#         except:
#             self.h2o_factor = 1.0

#         # Namelist options related to insolation
#         try:
#             self.dyofyr = namelist['radiation']['dyofyr']
#         except:
#             self.dyofyr = 0
#         try:
#             self.adjes = namelist['radiation']['adjes']
#         except:
#             print('Insolation adjustive factor not set so RadiationRRTM takes default value: adjes = 0.5 (12 hour of daylight).')
#             self.adjes = 0.5

#         try:
#             self.scon = namelist['radiation']['solar_constant']
#         except:
#             print('Solar Constant not set so RadiationRRTM takes default value: scon = 1360.0 .')
#             self.scon = 1360.0
#         try:
#             self.toa_sw = namelist['radiation']['toa_sw']
#         except:
#             print('TOA shortwave not set so RadiationRRTM takes default value: toa_sw = 420.0 .')
#             self.toa_sw = 420.0
#         if self.read_file:
#             rdr = cfreader(self.file, self.site)
#             self.toa_sw = rdr.get_timeseries_mean('swdn_toa')

#         try:
#             self.coszen = namelist['radiation']['coszen']
#         except:
#             if (self.toa_sw > 0.0):
#                 self.coszen = self.toa_sw / self.scon
#             else:
#                 print('Mean Daytime cos(SZA) not set so RadiationRRTM takes default value: coszen = 2.0/pi .')
#                 self.coszen = 2.0/pi

#         try:
#             self.adif = namelist['radiation']['adif']
#         except:
#             print('Surface diffusive albedo not set so RadiationRRTM takes default value: adif = 0.06 .')
#             self.adif = 0.06

#         try:
#             self.adir = namelist['radiation']['adir']
#         except:
#             if (self.coszen > 0.0):
#                 self.adir = (.026/(self.coszen**1.7+.065) + (.15*(self.coszen-0.10)*(self.coszen-0.50)*(self.coszen-1.00)))
#             else:
#                 self.adir = 0.0
#             print('Surface direct albedo not set so RadiationRRTM computes value: adif = %5.4f .'%(self.adir))

#         try:
#             self.uniform_reliq = namelist['radiation']['uniform_reliq']
#         except:
#             print('uniform_reliq not set so RadiationRRTM takes default value: uniform_reliq = False.')
#             self.uniform_reliq = False

#         try:
#             self.radiation_frequency = namelist['radiation']['frequency']
#         except:
#             print('radiation_frequency not set so RadiationRRTM takes default value: radiation_frequency = 0.0 (compute at every step).')
#             self.radiation_frequency = 60.0

#         self.next_radiation_calculate = 0.0

#         try:
#             self.IsdacCC_dT = namelist['initial']['dSST'] + namelist['initial']['dTi'] - 5.0
#             print('IsdacCC case: RRTM profiles are shifted according to %2.2f temperature change.'%(self.IsdacCC_dT))
#         except:
#             self.IsdacCC_dT = 0.0

#         return

#     cpdef initialize_io(self, NetCDFIO_Stats Stats):
#         Stats.add_global_mean('global_mean_dQTdt')
#         Stats.add_surface_global_mean('global_mean_RainRate')
#         Stats.add_zonal_mean('zonal_mean_dQTdt')
#         Stats.add_meridional_mean('meridional_mean_dQTdt')
#         Stats.add_surface_zonal_mean('zonal_mean_RainRate')
#         Stats.add_global_mean('global_mean_qv_star')
#         Stats.add_zonal_mean('zonal_mean_qv_star')
#         Stats.add_meridional_mean('meridional_mean_qv_star')
#         return

#     @cython.wraparound(False)
#     @cython.boundscheck(False)
#     @cython.cdivision(True)
#     cpdef update(self, Parameters Pr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS):
#         cdef:
#             Py_ssize_t nx = Pr.nlats
#             Py_ssize_t ny = Pr.nlons
#             Py_ssize_t nl = Pr.n_layers

#         return

#     cpdef stats_io(self, PrognosticVariables PV, NetCDFIO_Stats Stats):
#         Stats.write_global_mean('global_mean_dQTdt', PV.QT.mp_tendency)
#         Stats.write_surface_global_mean('global_mean_RainRate', self.RainRate)
#         Stats.write_zonal_mean('zonal_mean_dQTdt',PV.QT.mp_tendency)
#         Stats.write_meridional_mean('meridional_mean_dQTdt',PV.QT.mp_tendency)
#         Stats.write_surface_zonal_mean('zonal_mean_RainRate',self.RainRate)
#         Stats.write_global_mean('global_mean_qv_star',self.qv_star)
#         Stats.write_zonal_mean('zonal_mean_qv_star',self.qv_star)
#         Stats.write_meridional_mean('meridional_mean_qv_star',self.qv_star)
#         return

#     cpdef io(self, Parameters Pr, TimeStepping TS, NetCDFIO_Stats Stats):
#         Stats.write_2D_variable(Pr, int(TS.t) , 'Rain_Rate',self.RainRate)
#         return
