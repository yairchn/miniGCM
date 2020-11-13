import cython
from Grid cimport Grid
from math import *
import netCDF4 as nc
import numpy as np
import os
import time
import shutil
import sys

cdef class NetCDFIO_Stats:
    def __init__(self, namelist, Grid):
        self.root_grp = None
        self.variable_grp = None
        self.meridional_mean_grp = None
        self.zonal_mean_grp = None
        self.global_mean_grp = None

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        self.stats_frequency = namelist['io']['stats_frequency']
        self.output_frequency = namelist['io']['output_frequency']

        # Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                                   + self.uuid[len(self.uuid )-5:len(self.uuid)]))

        try:
            os.mkdir(outpath)
        except:
            pass

        self.path_plus_var = str(outpath + '/' + 'Fields/')
        try:
            os.mkdir(self.path_plus_var)
        except:
            pass

        self.stats_path = str( os.path.join(outpath, namelist['io']['stats_dir']))

        try:
            os.mkdir(self.stats_path)
        except:
            pass


        i = 1
        self.path_plus_file = str(self.stats_path + '/' + 'Stats.' + namelist['meta']['simname'] + '.nc')
        if os.path.exists(self.path_plus_file):
            res_name = 'Restart_'+str(i)
            print("Here " + res_name)
            self.path_plus_file = str(self.stats_path
                    + '/' + 'Stats.' + namelist['meta']['simname']
                    + '.' + res_name + '.nc')
            while os.path.exists(self.path_plus_file):
                i += 1
                res_name = 'Restart_'+str(i)
                print("Here " + res_name)
                self.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname']
                           + '.' + res_name + '.nc')


        shutil.copyfile(os.path.join( './', namelist['meta']['simname'] + '.in'),
                        os.path.join( outpath, namelist['meta']['simname'] + '.in'))
        self.setup_stats_file(Grid)
        return

    cpdef open_files(self):
        self.root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        self.global_mean_grp = self.root_grp.groups['global_mean']
        self.zonal_mean_grp = self.root_grp.groups['zonal_mean']
        self.meridional_mean_grp = self.root_grp.groups['meridional_mean']
        return

    cpdef close_files(self):
        self.root_grp.close()
        return

    cpdef setup_stats_file(self, Grid Gr):
        root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')

        # Set coordinates
        coordinate_grp = root_grp.createGroup('coordinates')
        coordinate_grp.createDimension('lat', Gr.nlats)
        coordinate_grp.createDimension('lon', Gr.nlons)
        coordinate_grp.createDimension('lay', 1)

        latitude = coordinate_grp.createVariable('latitude', 'f8',  ('lat', 'lon'), fill_value=0.0)
        latitude[:,:] = np.array(Gr.latitude)
        latitude_list = coordinate_grp.createVariable('latitude_list', 'f8', ('lat'), fill_value=0.0)
        latitude_list[:] = np.array(Gr.latitude_list)
        longitude = coordinate_grp.createVariable('longitude', 'f8', ('lat', 'lon'), fill_value=0.0)
        longitude[:,:] = np.array(Gr.longitude)
        longitude_list = coordinate_grp.createVariable('longitude_list', 'f8', ('lon') ,fill_value=0.0)
        longitude_list[:] = np.array(Gr.longitude_list)
        layer = coordinate_grp.createVariable('layers', 'f8',('lay'), fill_value=0.0)
        layer[:] = np.array(Gr.n_layers)
        del latitude, latitude_list, longitude, longitude_list, layer

        # Set maridional mean
        meridional_mean_grp = root_grp.createGroup('meridional_mean')
        meridional_mean_grp.createDimension('time', None)
        meridional_mean_grp.createDimension('lon', Gr.nlons)
        meridional_mean_grp.createDimension('lay',  Gr.n_layers)
        t = meridional_mean_grp.createVariable('t', 'f8', ('time'), fill_value=0.0)

        # Set zonal mean
        zonal_mean_grp = root_grp.createGroup('zonal_mean')
        zonal_mean_grp.createDimension('time', None)
        zonal_mean_grp.createDimension('lat', Gr.nlats)
        zonal_mean_grp.createDimension('lay',  Gr.n_layers)
        t = zonal_mean_grp.createVariable('t', 'f8',  ('time'), fill_value=0.0)

        # Set global_mean
        global_mean_grp = root_grp.createGroup('global_mean')
        global_mean_grp.createDimension('time', None)
        global_mean_grp.createDimension('lay',  Gr.n_layers)
        t = global_mean_grp.createVariable('t', 'f8', ('time'), fill_value=0.0)
        root_grp.close()
        return

    cpdef add_global_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        global_grp = root_grp.groups['global_mean']
        new_var = global_grp.createVariable(var_name, 'f8',('time', 'lay'), fill_value=0.0)
        root_grp.close()
        return

    cpdef add_meridional_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        meridional_mean_grp = root_grp.groups['meridional_mean']
        new_var = meridional_mean_grp.createVariable(var_name, 'f8', ('time','lon','lay'), fill_value=0.0)
        root_grp.close()
        return

    cpdef add_zonal_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        zonal_mean_grp = root_grp.groups['zonal_mean']
        new_var = zonal_mean_grp.createVariable(var_name, 'f8', ('time', 'lat','lay'), fill_value=0.0)
        root_grp.close()

        return

    cpdef write_global_mean(self, var_name, data, t):
        var = self.global_mean_grp.variables[var_name]
        var[-1,:] = np.array(np.mean(np.mean(data,0),0))
        return

    cpdef write_meridional_mean(self, var_name, data, t):
        var = self.meridional_mean_grp.variables[var_name]
        var[-1, :,:] = np.array(np.mean(data,0))
        return

    cpdef write_zonal_mean(self, var_name, data, t):
        var = self.zonal_mean_grp.variables[var_name]
        var[-1, :,:] = np.array(np.mean(data,1))
        return

    cpdef write_3D_variable(self, Gr, t, n_layers, var_name, data):
        root_grp = nc.Dataset(self.path_plus_var+var_name+'_'+str(t)+'.nc', 'w', format='NETCDF4')
        root_grp.createDimension('lat', Gr.nlats)
        root_grp.createDimension('lon', Gr.nlons)
        root_grp.createDimension('lay', n_layers)
        var = root_grp.createVariable(var_name, 'f8', ('lat', 'lon','lay'),fill_value=0.0)
        var[:,:,:] = np.array(data)
        root_grp.close()
        return

    cpdef write_simulation_time(self, t):
        # Write to global mean group
        global_mean_t = self.global_mean_grp.variables['t']
        global_mean_t[global_mean_t.shape[0]] = t

        # Write to zonal mean group
        zonal_mean_t = self.zonal_mean_grp.variables['t']
        zonal_mean_t[zonal_mean_t.shape[0]] = t

        # Write to meridional mean group
        meridional_mean_t = self.meridional_mean_grp.variables['t']
        meridional_mean_t[meridional_mean_t.shape[0]] = t
        return
