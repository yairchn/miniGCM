import numpy as np
import netCDF4 as nc
import argparse
import os
import json
import fnmatch

# command line:
# python machine_learning/make_training_date.py Output.HeldSuarez.theta7_efold/HeldSuarez.in
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("namelist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist

    # Setup the training output path
    uuid = str(namelist['meta']['uuid'])
    simulation_path = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                               + uuid[len(uuid )-12:len(uuid)],'Fields/'))
    output_path = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                               + uuid[len(uuid )-12:len(uuid)],'training/'))
    try:
        os.mkdir(output_path)
    except:
        pass

    varlist = {'Vorticity', 'Divergence', 'Temperature', 'Specific_humidity'}
    # define n_layers
    n_layers = np.shape(namelist['grid']['pressure_levels'])[0] - 1
    # open netdcf file
    root_grp = nc.Dataset(output_path+'Training_data.nc', 'w', format='NETCDF4')
    root_grp.createDimension('time', None)
    root_grp.createDimension('layer', n_layers)
    # root_grp.close()

    for varname in varlist:
        var = root_grp.createVariable(varname, 'f8', ('time','layer'))
        I = 0
        for filename in os.listdir(simulation_path):
            if filename.startswith(varname):
                data = nc.Dataset(simulation_path+filename, 'r')
                var_values = np.array(data.variables[varname])
                var_1D = var_values.reshape(-1,np.shape(var_values)[2])
                if I==0:
                    var_collecated=var_1D
                else:
                    var_collecated=np.vstack([var_1D, var_1D])
                I += 1

        var[:, :] = np.array(var_collecated)
        del var_collecated, var_values, data, var_1D

    varname = 'Pressure'
    root_grp.createDimension('s', 1)
    var_p = root_grp.createVariable(varname, 'f8', ('time','s'))
    J = 0
    for filename in os.listdir(simulation_path):
        if filename.startswith(varname):
            data = nc.Dataset(simulation_path+filename, 'r')
            var_values = np.array(data.variables[varname])
            var_1D = var_values.reshape(-1)
            if J==0:
                var_collecated=var_1D
            else:
                var_collecated=np.hstack([var_1D, var_1D])
            J += 1
    var_p[:] = np.array(var_collecated)
    del var_collecated

    root_grp.close()

if __name__ == '__main__':
    main()
