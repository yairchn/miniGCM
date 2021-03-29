import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
import os

cdef class Parameters:
    def __init__(self, namelist):

        self.thermodynamics_type = namelist['thermodynamics']['thermodynamics_type']
        if (self.thermodynamics_type=='dry'):
            self.moist_index = 0.0
        else:
            self.moist_index = 1.0

        self.case      = namelist['meta']['casename']
        self.lat       = namelist['grid']['degree_latitute']
        self.rsphere   = namelist['planet']['planet_radius']
        self.nx        = namelist['grid']['number_of_x_points']
        self.ny        = namelist['grid']['number_of_y_points']
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
        self.Coriolis  = 2.0*self.Omega*np.sin(self.lat)

        self.numerical_scheme      = namelist['grid']['numerical_scheme']

        self.efold        = namelist['diffusion']['e_folding_timescale']
        self.noise_amp    = namelist['initialize']['noise_amplitude']
        self.T1           = namelist['initialize']['T1'] = 229.0
        self.T2           = namelist['initialize']['T2'] = 259.0
        self.T3           = namelist['initialize']['T3'] = 291.0
        self.Hamp1        = namelist['initialize']['Tamp1'] = 0.2
        self.Hamp2        = namelist['initialize']['Tamp2'] = 1.0
        self.Hamp3        = namelist['initialize']['Tamp3'] = 0.0
        self.QT1          = namelist['initialize']['QT1'] = 2.5000e-04
        self.QT2          = namelist['initialize']['QT2'] = 0.0016
        self.QT3          = namelist['initialize']['QT3'] = 0.0115
        self.sigma_H      = namelist['initialize']['warm_core_width']
        self.amp_H        = namelist['initialize']['warm_core_amplitude']


        self.surface_model = namelist['surface']['surface_model']
        self.inoise        = namelist['initialize']['inoise']
        self.noise_amp     = namelist['initialize']['noise_amplitude']

        self.uuid        = namelist['meta']['uuid']
        self.casename    = namelist['meta']['casename']
        self.outpath     = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                            + self.uuid[len(self.uuid )-5:len(self.uuid)]))
        self.logfilename = self.outpath+'/'+self.casename+'.log'

        self.tau  = namelist['forcing']['relaxation_timescale']

        return

