import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
import shtns
import sphTrans as sph
import os

cdef class Parameters:
    def __init__(self, namelist):

        self.thermodynamics_type = namelist['thermodynamics']['thermodynamics_type']
        if (self.thermodynamics_type=='dry'):
            self.moist_index = 0.0
        else:
            self.moist_index = 1.0

        self.case      = namelist['meta']['casename']
        self.nlats     = namelist['grid']['number_of_latitute_points']
        self.nlons     = namelist['grid']['number_of_longitude_points']
        self.rsphere   = namelist['planet']['planet_radius']
        self.n_layers  = namelist['grid']['number_of_layers']
        self.p1        = namelist['grid']['p1']
        self.p2        = namelist['grid']['p2']
        self.p3        = namelist['grid']['p3']
        self.p_ref     = namelist['grid']['p_ref']
        self.Omega     = namelist['planet']['Omega_rotation']
        self.g         = namelist['planet']['gravity']
        self.cp        = namelist['thermodynamics']['heat_capacity']
        self.Rd        = namelist['thermodynamics']['dry_air_gas_constant']
        self.Rv        = namelist['thermodynamics']['vapor_gas_constant']
        self.Lv        = namelist['thermodynamics']['latent_heat_evap']
        self.qv_star0  = namelist['thermodynamics']['pv_star_triple_point']
        self.T_0       = namelist['thermodynamics']['triple_point_temp']
        self.eps_v     = self.Rd/self.Rv
        self.kappa     = self.Rd/self.cp

        self.dissipation_order = namelist['diffusion']['dissipation_order']
        self.truncation_order  = namelist['diffusion']['truncation_order']
        self.truncation_number = int(self.nlons/self.truncation_order)
        self.efold             = namelist['diffusion']['e_folding_timescale']
        self.noise_amp      = namelist['initialize']['noise_amplitude']

        self.surface_model = namelist['surface']['surface_model']
        self.inoise        = namelist['initialize']['inoise']

        self.uuid        = namelist['meta']['uuid']
        self.casename    = namelist['meta']['casename']
        self.outpath     = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                            + self.uuid[len(self.uuid )-5:len(self.uuid)]))
        self.logfilename = self.outpath+'/'+self.casename+'.log'

        return

