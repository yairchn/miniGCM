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
    namelist_defaults['diffusion']['dissipation_order'] = 8.0
    namelist_defaults['diffusion']['truncation_order'] = 4
    namelist_defaults['diffusion']['e_folding_timescale'] = 0.01

    namelist_defaults['grid'] = {}
    namelist_defaults['grid']['dims'] = 1
    namelist_defaults['grid']['gw'] = 2
    namelist_defaults['grid']['number_of_latitute_points'] =  256
    namelist_defaults['grid']['number_of_longitude_points'] = 512
    namelist_defaults['grid']['number_of_layers'] =  3
    namelist_defaults['grid']['rho3']       =  1000.0*1.0  # [km/m^3]
    namelist_defaults['grid']['rho2']       =  1000.0*0.95 # [km/m^3]
    namelist_defaults['grid']['rho1']       =  1000.0*0.9  # [km/m^3]

    namelist_defaults['planet'] = {}
    namelist_defaults['planet']['planet_radius']    = 6.37122e6 # earth radius [m]
    namelist_defaults['planet']['Omega_rotation']   = 7.292e-5  # rotation rate [1/s]
    namelist_defaults['planet']['gravity']          = 9.80616   # gravity [m/s**2]

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

    if case_name == 'Defualt':
        namelist = Defualt(namelist_defaults)
    else:
        print('Not a valid case name')
        exit()

    write_file(namelist)


def Defualt(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['meta']['simname'] = 'Defualt'
    namelist['meta']['casename'] = 'Defualt'

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['thermodynamics_type'] = 'dry'
    namelist['thermodynamics']['dry_air_gas_constant'] = 287.04     # [J / (kg K)]
    namelist['thermodynamics']['vapor_gas_constant']   = 461.5     # [J / (kg K)]
    namelist['thermodynamics']['latent_heat_evap']     = 2.5008e6  # [J / (kg K)]
    namelist['thermodynamics']['pv_star_triple_point'] = 610.78
    namelist['thermodynamics']['triple_point_temp']    = 273.16    # [K]

    namelist['microphysics'] = {}
    namelist['microphysics']['rain_model'] = 'None'

    namelist['forcing']['forcing_type'] = 'HeldSuarez'
    namelist['forcing']['k_a']          = 1./40.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_s']          =  1./4.0/(24.0*3600.0)  # [1/sec]
    namelist['forcing']['k_f']          = 1.0/(24.0*3600.0)      # [1/sec]
    namelist['forcing']['k_b']          = 1./40.0/(24.0*3600.0)/2.0  # [1/sec]
    namelist['forcing']['equator_to_pole_dH']         = 60.                    # Characteristic temperature change in meridional direction [K]
    namelist['forcing']['equatorial_depth']    = 315.                   # Characteristic temperature at the equator [K]
    namelist['forcing']['lapse_rate']   = 10.0                   # Characteristic potential temperature change in vertical [K]

    namelist['microphysics']['rain_model'] = 'None'

    namelist['surface'] = {}
    namelist['surface']['surface_model'] = 'bulk_formula'
    namelist['surface']['momentum_transfer_coeff'] = 0.00044
    namelist['surface']['sensible_heat_transfer_coeff'] = 0.00044
    namelist['surface']['latent_heat_transfer_coeff'] = 0.00044
    namelist['surface']['surface_temp_diff'] = 29.0 # [K]
    namelist['surface']['surface_temp_min'] = 271.0 # [K]
    namelist['surface']['surface_temp_lat_dif'] = 26.0*3.14/180.0

    namelist['diffusion']['type'] = 'hyperdiffusion'
    namelist['diffusion']['order'] = 8.0

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
