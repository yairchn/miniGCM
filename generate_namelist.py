import argparse
import json
import pprint
from sys import exit
import uuid
import ast
import copy

def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')

    args = parser.parse_args()

    case_name = args.case_name

    namelist_defaults = {}
    namelist_defaults['timestepping'] = {}
    namelist_defaults['timestepping']['CFL_limit'] = 0.8
    namelist_defaults['timestepping']['dt'] = 100.0
    namelist_defaults['timestepping']['t_max'] = 100.0 # days

    namelist_defaults['forcing'] = {}

    namelist_defaults['diffusion'] = {}
    namelist_defaults['diffusion']['e_folding_timescale'] = 600

    namelist_defaults['grid'] = {}
    namelist_defaults['grid']['dims'] = 1
    namelist_defaults['grid']['gw'] = 2
    namelist_defaults['grid']['number_of_x_points'] = 100
    namelist_defaults['grid']['number_of_y_points'] = 100
    namelist_defaults['grid']['x_resolution'] = 10000
    namelist_defaults['grid']['y_resolution'] = 10000
    namelist_defaults['grid']['number_of_layers'] =  3


    namelist_defaults['grid']['H3']       =  30.0  # [pasc]
    namelist_defaults['grid']['H2']       =  30.0  # [pasc]
    namelist_defaults['grid']['H1']       =  30.0  # [pasc]
    namelist_defaults['grid']['degree_latitute'] = 20.0
    namelist_defaults['grid']['numerical_scheme'] = 'centeral_differneces'

    namelist_defaults['planet'] = {}
    namelist_defaults['planet']['planet_radius']    = 6.37122e6 # earth radius [m]
    namelist_defaults['planet']['Omega_rotation']   = 7.292e-5  # rotation rate [1/s]
    namelist_defaults['planet']['gravity']          = 9.80616   # gravity [m/s**2]

    namelist_defaults['thermodynamics'] = {}
    namelist_defaults['thermodynamics']['heat_capacity']        = 1004.0    # [J / (kg K)]
    namelist_defaults['thermodynamics']['dry_air_gas_constant'] = 287.04     # [J / (kg K)]
    namelist_defaults['thermodynamics']['vapor_gas_constant']   = 461.5     # [J / (kg K)]
    namelist_defaults['thermodynamics']['latent_heat_evap']     = 2.5008e6  # [J / (kg K)]
    namelist_defaults['thermodynamics']['pv_star_triple_point'] = 610.78
    namelist_defaults['thermodynamics']['triple_point_temp']    = 273.16    # [K]

    namelist_defaults['initialize'] = {}
    namelist_defaults['initialize']['inoise']                   = 1 # flag for noise in initial condition
    namelist_defaults['initialize']['noise_amplitude']          = 0.01 # amplitude of initial noise in K

    namelist_defaults['output'] = {}
    namelist_defaults['output']['output_root'] = './'

    namelist_defaults['io'] = {}
    namelist_defaults['io']['stats_dir'] = 'stats'
    namelist_defaults['io']['stats_frequency'] = 1.0
    namelist_defaults['io']['output_frequency'] = 7.0

    namelist_defaults['meta'] = {}

    if case_name == 'DryVortex':
        namelist = DryVortex(namelist_defaults)
    elif case_name == 'MoistVortex':
        namelist = MoistVortex(namelist_defaults)
    elif case_name == 'ReedJablonowski':
        namelist = ReedJablonowski(namelist_defaults)
    else:
        print('Not a valid case name')
        exit()

    write_file(namelist)


def DryVortex(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['meta']['simname'] = 'DryVortex'
    namelist['meta']['casename'] = 'DryVortex'

    namelist['thermodynamics']['thermodynamics_type'] = 'Moist'

    namelist['microphysics'] = {}
    namelist['microphysics']['rain_model'] = 'None'

    namelist['forcing']['forcing_type'] = 'BetterMiller'
    namelist['forcing']['relaxation_timescale'] = 3.0*3600.0
    namelist['initialize']['warm_core_width']  = 100000.0
    namelist['initialize']['warm_core_amplitude']  = 5.0
    namelist['forcing']['k_a']          = 1./40.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_s']          =  1./4.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_f']          = 1.0/(24.0*3600.0)      # [1/sec]
    namelist['forcing']['k_b']          = 1./40.0/(24.0*3600.0)/2.0  # [1/sec]
    namelist['forcing']['equatorial_temperature']    = 315.                   # Characteristic temperature at the equator [K]
    namelist['forcing']['lapse_rate']   = 10.0                   # Characteristic potential temperature change in vertical [K]

    namelist['microphysics']['rain_model'] = 'None'

    namelist['surface'] = {}
    namelist['surface']['surface_model'] = 'None'

    namelist['diffusion']['type'] = 'hyperdiffusion'
    namelist['diffusion']['order'] = 8.0

    return namelist

def MoistVortex(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['meta']['simname'] = 'MoistVortex'
    namelist['meta']['casename'] = 'MoistVortex'

    namelist['thermodynamics']['thermodynamics_type'] = 'Moist'

    namelist['microphysics'] = {}
    namelist['microphysics']['rain_model'] = 'None'

    namelist['forcing']['forcing_type'] = 'BetterMiller'
    namelist['forcing']['relaxation_timescale'] = 3.0*3600.0
    namelist['initialize']['warm_core_width']  = 100000.0
    namelist['initialize']['warm_core_amplitude']  = 5.0
    namelist['forcing']['k_a']          = 1./40.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_s']          =  1./4.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_f']          = 1.0/(24.0*3600.0)      # [1/sec]
    namelist['forcing']['k_b']          = 1./40.0/(24.0*3600.0)/2.0  # [1/sec]
    namelist['forcing']['equatorial_temperature']    = 315.                   # Characteristic temperature at the equator [K]
    namelist['forcing']['lapse_rate']   = 10.0                   # Characteristic potential temperature change in vertical [K]

    namelist['initialize']['Hamp1'] = 0.2
    namelist['initialize']['Hamp2'] = 1.0
    namelist['initialize']['Hamp3'] = 0.0
    namelist['initialize']['QT1'] = 2.5000e-04
    namelist['initialize']['QT2'] = 0.0016
    namelist['initialize']['QT3'] = 0.0115

    namelist['microphysics']['rain_model'] = 'None'

    namelist['surface'] = {}
    namelist['surface']['surface_model'] = 'None'

    namelist['diffusion']['type'] = 'hyperdiffusion'
    namelist['diffusion']['order'] = 8.0

    return namelist

def ReedJablonowski(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['number_of_layers'] = 3
    namelist['grid']['number_of_x_points'] = 100
    namelist['grid']['number_of_y_points'] = 100

    namelist['timestepping']['dt'] = 100.0 # sec
    namelist['timestepping']['t_max'] = 100.0 # days

    namelist['meta']['simname'] = 'ReedJablonowski'
    namelist['meta']['casename'] = 'ReedJablonowski'

    namelist['forcing']['forcing_type'] = 'ReedJablonowski'
    namelist['forcing']['sigma_b']      = 0.7                    # sigma coordiantes as sigma=p/ps
    namelist['forcing']['k_a']          = 1.0/40.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_s']          = 1.0/4.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_f']          = 1.0/(24.0*3600.0)      # [1/sec]
    namelist['forcing']['k_b']          = 1.0/40.0/(24.0*3600.0)/2.0  # [1/sec]
    namelist['forcing']['equatorial_temperature'] = 294.0 # Surface temperature at the equator [K]
    namelist['forcing']['polar_temperature']      = 240.0 # Surface temperature at the pole [K]
    namelist['forcing']['equator_to_pole_dT']     = 65.0  # Characteristic temperature change in meridional direction [K]
    namelist['forcing']['lapse_rate']             = 10.0 # Characteristic potential temperature change in vertical [K]
    namelist['forcing']['initial_profile_power'] = 3.0
    namelist['forcing']['initial_surface_qt'] = 0.018

    # # from paper
    # namelist['initialize']['Tropopause height'] = 15000
    # namelist['initialize']['Specific humidity at the surface'] = 21
    # namelist['initialize']['Specific humidity of upper atmosphere'] = 1028
    # namelist['initialize']['Constant 1 for specific humidity profile'] = 3000
    # namelist['initialize']['Constant 2 for specific humidity profile'] = 8000
    # namelist['initialize']['Surface temperature and sea surface temperature'] = 302.15
    # namelist['initialize']['Background virtual temperature at the surface'] = T0(1 1 0.608q0)
    # namelist['initialize']['Virtual temperature in upper atmosphere'] = Ty02Gzt
    # namelist['initialize']['Virtual temperature lapse rate'] = 0.007
    # namelist['initialize']['Background surface pressure'] = 1015
    # namelist['initialize']['Pressure at the tropopause height zt'] = See Eq. (5)
    # namelist['initialize']['Surface pressure difference between the background and center'] = 11.15
    # namelist['initialize']['Constant 1 for pressure fit'] = 282
    # namelist['initialize']['Constant 2 for pressure fit'] = 7000
    # namelist['initialize']['Center latitude of initial vortex'] = 10
    # namelist['initialize']['Center longitude of the initial vortex'] = 180
    # # check these 
    # # namelist['initialize']['Small constant to avoid division by zero'] = 2V sin(uc) 10225
    # # namelist['initialize']['Convergence limit for fixed-point iterations'] = 2 3 10213


    namelist['thermodynamics']['thermodynamics_type'] = 'moist'

    namelist['microphysics'] = {}
    namelist['microphysics']['rain_model'] = 'Kessler_cutoff'
    namelist['microphysics']['water_density'] = 1000.0            # g/m^3
    namelist['microphysics']['Magnus_formula_A'] = 6.1094
    namelist['microphysics']['Magnus_formula_B'] = 17.625
    namelist['microphysics']['Magnus_formula_C'] = 243.04
    namelist['microphysics']['molar_mass_ratio'] = 1.60745384883
    namelist['microphysics']['max_supersaturation'] = 0.0

    namelist['surface'] = {}
    namelist['surface']['surface_model'] = 'bulk_formula'
    namelist['surface']['momentum_transfer_coeff']      = 0.0044
    namelist['surface']['sensible_heat_transfer_coeff'] = 0.0044
    namelist['surface']['latent_heat_transfer_coeff']   = 0.0044
    namelist['surface']['sea_surface_temperature']      = 300.0 # [K]

    return namelist

def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()
