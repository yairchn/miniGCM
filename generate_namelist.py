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
    namelist_defaults['timestepping']['CFL_limit'] = 0.5
    namelist_defaults['timestepping']['dt'] = 1000.0

    namelist_defaults['forcing'] = {}

    namelist_defaults['diffusion'] = {}
    namelist_defaults['diffusion']['dissipation_order'] = 8.0
    namelist_defaults['diffusion']['truncation_order'] = 3
    namelist_defaults['diffusion']['e_folding_timescale'] = 600.0

    namelist_defaults['grid'] = {}
    namelist_defaults['grid']['dims'] = 1
    namelist_defaults['grid']['gw'] = 2
    namelist_defaults['grid']['number_of_latitute_points'] =  256
    namelist_defaults['grid']['number_of_longitude_points'] = 512
    namelist_defaults['grid']['number_of_layers'] =  3
    namelist_defaults['grid']['p3']       =  850.0*1.e2  # [pasc]
    namelist_defaults['grid']['p2']       =  500.0*1.e2  # [pasc]
    namelist_defaults['grid']['p1']       =  250.0*1.e2  # [pasc]
    namelist_defaults['grid']['p_ref']    =  1000.0*1.e2 # [pasc]

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

    namelist_defaults['output'] = {}
    namelist_defaults['output']['output_root'] = './'

    namelist_defaults['io'] = {}
    namelist_defaults['io']['stats_dir'] = 'stats'
    namelist_defaults['io']['stats_frequency'] = 1.0
    namelist_defaults['io']['output_frequency'] = 7.0

    namelist_defaults['meta'] = {}

    if case_name == 'HeldSuarez':
        namelist = HeldSuarez(namelist_defaults)
    elif case_name == 'HeldSuarezMoist':
        namelist = HeldSuarezMoist(namelist_defaults)
    else:
        print('Not a valid case name')
        exit()

    write_file(namelist)


def HeldSuarez(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['timestepping']['dt'] = 100.0 # sec
    namelist['timestepping']['t_max'] = 100.0 # days

    namelist['meta']['simname'] = 'HeldSuarez'
    namelist['meta']['casename'] = 'HeldSuarez'

    namelist['thermodynamics']['thermodynamics_type'] = 'dry'

    namelist['microphysics'] = {}
    namelist['microphysics']['rain_model'] = 'None'

    namelist['forcing']['forcing_type'] = 'HeldSuarez'
    namelist['forcing']['sigma_b']      = 0.7                    # sigma coordiantes as sigma=p/ps
    namelist['forcing']['k_a']          = 1./40.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_s']          =  1./4.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_f']          = 1.0/(24.0*3600.0)      # [1/sec]
    namelist['forcing']['k_b']          = 1./40.0/(24.0*3600.0)/2.0  # [1/sec]
    namelist['forcing']['equator_to_pole_dT']         = 60.                    # Characteristic temperature change in meridional direction [K]
    namelist['forcing']['equatorial_temperature']    = 300.                   # Characteristic temperature at the equator [K]
    namelist['forcing']['lapse_rate']   = 10.0                   # Characteristic potential temperature change in vertical [K]
    namelist['forcing']['relaxation_temperature'] = 300.         # mean temp (some typical range) [K]

    namelist['microphysics']['rain_model'] = 'None'

    namelist['surface'] = {}
    namelist['surface']['surface_model'] = 'None'

    namelist['diffusion']['type'] = 'hyperdiffusion'

    return namelist

def HeldSuarezMoist(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['n_layers'] = 3
    namelist['grid']['nx'] = 3
    namelist['grid']['ny'] = 3

    namelist['timestepping']['dt'] = 500.0 # sec
    namelist['timestepping']['t_max'] = 100.0 # days

    namelist['meta']['simname'] = 'HeldSuarezMoist'
    namelist['meta']['casename'] = 'HeldSuarezMoist'

    namelist['forcing']['forcing_type'] = 'HeldSuarezMoist'
    namelist['forcing']['sigma_b']      = 0.7                    # sigma coordiantes as sigma=p/ps
    namelist['forcing']['k_a']          = 1.0/40.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_s']          = 1.0/4.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_f']          = 1.0/(24.0*3600.0)      # [1/sec]
    namelist['forcing']['k_b']          = 1.0/40.0/(24.0*3600.0)/2.0  # [1/sec]
    namelist['forcing']['equatorial_temperature'] = 310.0 # Surface temperature at the equator [K]
    namelist['forcing']['polar_temperature']      = 240.0 # Surface temperature at the pole [K]
    namelist['forcing']['equator_to_pole_dT']     = 65.0  # Characteristic temperature change in meridional direction [K]
    namelist['forcing']['lapse_rate']             = 10.0 # Characteristic potential temperature change in vertical [K]
    namelist['forcing']['relaxation_temperature'] = 300.         # mean temp (some typical range) [K]
    namelist['forcing']['initial_profile_power'] = 3.0
    namelist['forcing']['initial_surface_qt'] = 0.018
    namelist['forcing']['Gamma_init'] = 0.005


    namelist['thermodynamics']['thermodynamics_type'] = 'moist'
    namelist['thermodynamics']['verical_half_width_of_the_q'] = 34000.0 # pasc
    namelist['thermodynamics']['horizontal_half_width_of_the_q'] = 0.6981 # radians lat

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
    namelist['surface']['momentum_transfer_coeff'] = 0.0044
    namelist['surface']['sensible_heat_transfer_coeff'] = 0.0044
    namelist['surface']['latent_heat_transfer_coeff'] = 0.0044
    namelist['surface']['surface_temp_diff'] = 29.0 # [K]
    namelist['surface']['surface_temp_min'] = 271.0 # [K]
    namelist['surface']['surface_temp_lat_dif'] = 26.0*3.14/180.0

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
