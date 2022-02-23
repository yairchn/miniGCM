import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import netCDF4 as nc
import numpy as np
cimport numpy as np
import os
import shutil
from Parameters cimport Parameters

cdef class NetCDFIO_Stats:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.root_grp = None
        self.variable_grp = None
        self.meridional_mean_grp = None
        self.zonal_mean_grp = None
        self.global_mean_grp = None
        self.surface_zonal_mean_grp = None

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        self.stats_frequency = namelist['io']['stats_frequency']*24.0*3600.0 # sec/day
        self.output_frequency = namelist['io']['output_frequency']*24.0*3600.0 # sec/day

        # Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                                   + self.uuid[len(self.uuid )-12:len(self.uuid)]))

        try:
            os.mkdir(outpath)
        except:
            pass

        Pr.input_folder = str(outpath + '/' + 'Fields/')
        if Pr.restart:
            self.output_folder = str(outpath + '/' + 'Fields_restart/')
        else:
            self.output_folder = Pr.input_folder
        try:
            os.mkdir(Pr.input_folder)
        except:
            pass

        try:
            os.mkdir(self.output_folder)
        except:
            pass

        self.stats_path = str( os.path.join(outpath, namelist['io']['stats_dir']))

        try:
            os.mkdir(self.stats_path)
        except:
            pass


        i = 1
        Pr.path_plus_file = str(self.stats_path + '/' + 'Stats.' + namelist['meta']['simname'] + '.nc')
        if Pr.restart:
            Pr.path_source_file = Pr.path_plus_file

        if Pr.restart:
            if os.path.exists(Pr.path_plus_file):
                res_name = 'Restart_'+str(i)
                print("Here " + res_name)
                Pr.path_plus_file = str(self.stats_path
                        + '/' + 'Stats.' + namelist['meta']['simname']
                        + '.' + res_name + '.nc')
                while os.path.exists(Pr.path_plus_file):
                    i += 1
                    res_name = 'Restart_'+str(i)
                    print("Here " + res_name)
                    Pr.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname']
                               + '.' + res_name + '.nc')
        else:
            if os.path.exists(Pr.path_plus_file):
                res_name = 'Rerun_'+str(i)
                print("Here " + res_name)
                Pr.path_plus_file = str(self.stats_path
                        + '/' + 'Stats.' + namelist['meta']['simname']
                        + '.' + res_name + '.nc')
                while os.path.exists(Pr.path_plus_file):
                    i += 1
                    res_name = 'Rerun_'+str(i)
                    print("Here " + res_name)
                    Pr.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname']
                               + '.' + res_name + '.nc')


        shutil.copyfile(os.path.join( './', namelist['meta']['simname'] + '.in'),
                        os.path.join( outpath, namelist['meta']['simname'] + '.in'))
        self.path_plus_file = Pr.path_plus_file
        self.setup_stats_file(Pr, Gr)

    cpdef open_files(self):
        self.root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        self.global_mean_grp = self.root_grp.groups['global_mean']
        self.zonal_mean_grp = self.root_grp.groups['zonal_mean']
        self.meridional_mean_grp = self.root_grp.groups['meridional_mean']
        self.surface_zonal_mean_grp = self.root_grp.groups['surface_zonal_mean']
        self.surface_meridional_mean_grp = self.root_grp.groups['surface_meridional_mean']
        return

    cpdef close_files(self):
        self.root_grp.close()
        return

    cpdef setup_stats_file(self, Parameters Pr, Grid Gr):

        root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')

        # Set coordinates
        coordinate_grp = root_grp.createGroup('coordinates')
        coordinate_grp.createDimension('lat', Pr.nlats)
        coordinate_grp.createDimension('lon', Pr.nlons)
        coordinate_grp.createDimension('lay', Pr.n_layers+1)

        latitude = coordinate_grp.createVariable('latitude', 'f8',  ('lat', 'lon'))
        latitude[:,:] = np.array(Gr.latitude)
        latitude_list = coordinate_grp.createVariable('latitude_list', 'f8', ('lat'))
        latitude_list[:] = np.array(Gr.latitude_list)
        longitude = coordinate_grp.createVariable('longitude', 'f8', ('lat', 'lon'))
        longitude[:,:] = np.array(Gr.longitude)
        longitude_list = coordinate_grp.createVariable('longitude_list', 'f8', ('lon'))
        longitude_list[:] = np.array(Gr.longitude_list)
        pressure_levels = coordinate_grp.createVariable('pressure_levels', 'f8',('lay'))
        pressure_levels[:] = np.array(Pr.pressure_levels.base)
        del latitude, latitude_list, longitude, longitude_list, pressure_levels

        # Set maridional mean
        meridional_mean_grp = root_grp.createGroup('meridional_mean')
        meridional_mean_grp.createDimension('time', None)
        meridional_mean_grp.createDimension('lon', Pr.nlons)
        meridional_mean_grp.createDimension('lay',  Pr.n_layers)
        t = meridional_mean_grp.createVariable('t', 'f8', ('time'))

        # Set zonal mean
        zonal_mean_grp = root_grp.createGroup('zonal_mean')
        zonal_mean_grp.createDimension('time', None)
        zonal_mean_grp.createDimension('lat', Pr.nlats)
        zonal_mean_grp.createDimension('lay',  Pr.n_layers)
        t = zonal_mean_grp.createVariable('t', 'f8',  ('time'))

        # Set global_mean
        global_mean_grp = root_grp.createGroup('global_mean')
        global_mean_grp.createDimension('time', None)
        global_mean_grp.createDimension('lay',  Pr.n_layers)
        t = global_mean_grp.createVariable('t', 'f8', ('time'))

        # Set surface_global_mean
        surface_global_mean_grp = root_grp.createGroup('surface_global_mean')
        surface_global_mean_grp.createDimension('time', None)
        t = surface_global_mean_grp.createVariable('t', 'f8', ('time'))

        # Set surface_meridional_mean
        surface_meridional_mean_grp = root_grp.createGroup('surface_meridional_mean')
        surface_meridional_mean_grp.createDimension('time', None)
        surface_meridional_mean_grp.createDimension('lon', Pr.nlons)
        t = surface_meridional_mean_grp.createVariable('t', 'f8', ('time'))

        # Set surface_zonal_mean
        surface_zonal_mean_grp = root_grp.createGroup('surface_zonal_mean')
        surface_zonal_mean_grp.createDimension('time', None)
        surface_zonal_mean_grp.createDimension('lat', Pr.nlats)
        t = surface_zonal_mean_grp.createVariable('t', 'f8', ('time'))
        root_grp.close()

        return

    cpdef add_global_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        global_grp = root_grp.groups['global_mean']
        new_var = global_grp.createVariable(var_name, 'f8',('time', 'lay'))
        root_grp.close()
        return

    cpdef add_meridional_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        meridional_mean_grp = root_grp.groups['meridional_mean']
        new_var = meridional_mean_grp.createVariable(var_name, 'f8', ('time','lon','lay'))
        root_grp.close()
        return

    cpdef add_surface_global_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        global_grp = root_grp.groups['global_mean']
        new_var = global_grp.createVariable(var_name, 'f8',('time'))
        root_grp.close()
        return

    cpdef add_surface_zonal_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        surface_zonal_mean_grp = root_grp.groups['surface_zonal_mean']
        new_var = surface_zonal_mean_grp.createVariable(var_name, 'f8', ('time','lat'))
        root_grp.close()
        return

    cpdef add_surface_meridional_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        surface_meridional_mean_grp = root_grp.groups['surface_meridional_mean']
        new_var = surface_meridional_mean_grp.createVariable(var_name, 'f8', ('time','lon'))
        root_grp.close()
        return

    cpdef add_zonal_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        zonal_mean_grp = root_grp.groups['zonal_mean']
        new_var = zonal_mean_grp.createVariable(var_name, 'f8', ('time', 'lat','lay'))
        root_grp.close()
        return

    cpdef write_global_mean(self, Grid Gr, var_name, data):
        var = self.global_mean_grp.variables[var_name]
        zonal_mean = np.array(np.mean(data, axis = (1,2)))
        var[-1,:] = np.sum(np.multiply(Gr.lat_weights, zonal_mean))
        return

    cpdef write_zonal_mean(self, var_name, data):
        var = self.zonal_mean_grp.variables[var_name]
        var[-1, :,:] = np.array(np.mean(data,1))
        return

    cpdef write_meridional_mean(self, var_name, data):
        var = self.meridional_mean_grp.variables[var_name]
        var[-1,:,:] = np.array(np.mean(data,0))
        return

    cpdef write_surface_global_mean(self, Grid Gr, var_name, data):
        var = self.global_mean_grp.variables[var_name]
        zonal_mean = np.array(np.mean(data,1))
        var[-1] = np.sum(np.multiply(Gr.lat_weights, zonal_mean))
        return

    cpdef write_surface_zonal_mean(self, var_name, data):
        var = self.surface_zonal_mean_grp.variables[var_name]
        var[-1,:] = np.array(np.mean(data,1))
        return

    cpdef write_surface_meridional_mean(self, var_name, data):
        var = self.surface_meridional_mean_grp.variables[var_name]
        var[-1,:] = np.array(np.mean(data,0))
        return


    cpdef write_3D_variable(self, Parameters Pr, t, n_layers, var_name, data):
        root_grp = nc.Dataset(self.output_folder+var_name+'_'+str(t)+'.nc', 'w', format='NETCDF4')
        root_grp.createDimension('lat', Pr.nlats)
        root_grp.createDimension('lon', Pr.nlons)
        root_grp.createDimension('lay', n_layers)
        var = root_grp.createVariable(var_name, 'f8', ('lat', 'lon','lay'))
        var[:,:,:] = np.array(data)
        root_grp.close()
        return

    cpdef write_spectral_field(self, Parameters Pr, t, nlm, n_layers, var_name, data):
        root_grp = nc.Dataset(self.output_folder+var_name+'_'+str(t)+'.nc', 'w', format='NETCDF4')
        root_grp.createDimension('nlm', nlm)
        root_grp.createDimension('lay', n_layers)
        var = root_grp.createVariable(var_name, 'f8', ('nlm', 'lay'))
        var[:,:] = np.array(data)
        root_grp.close()
        return

    cpdef write_spectral_analysis(self, n_wavenumbers, n_layers, var_name, data):
        root_grp = nc.Dataset(self.stats_path+var_name+'.nc', 'w', format='NETCDF4')
        root_grp.createDimension('n_wn', n_wavenumbers)
        root_grp.createDimension('lay', n_layers)
        var = root_grp.createVariable(var_name, 'f8', ('n_wn', 'lay'))
        var[:,:] = np.array(data)
        root_grp.close()
        return

    cpdef write_2D_variable(self, Parameters Pr, t, var_name, data):
        root_grp = nc.Dataset(self.output_folder+var_name+'_'+str(t)+'.nc', 'w', format='NETCDF4')
        root_grp.createDimension('lat', Pr.nlats)
        root_grp.createDimension('lon', Pr.nlons)
        var = root_grp.createVariable(var_name, 'f8', ('lat', 'lon'))
        var[:,:] = np.array(data)
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

        surface_zonal_mean_grp_t = self.surface_zonal_mean_grp.variables['t']
        surface_zonal_mean_grp_t[surface_zonal_mean_grp_t.shape[0]] = t
        return
