import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import netCDF4 as nc
import numpy as np
cimport numpy as np
import os
import shutil
from Parameters cimport Parameters
from UtilityFunctions import axisymmetric_mean


cdef class NetCDFIO_Stats:
    def __init__(self, Parameters Pr, Grid Gr, namelist):
        self.xc = Gr.xc
        self.yc = Gr.yc
        self.nl = Gr.nl
        self.root_grp = None
        self.variable_grp = None
        self.axisymmetric_mean_grp = None
        self.global_mean_grp = None
        self.surface_axisymmetric_mean_grp = None

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        self.stats_frequency = namelist['io']['stats_frequency']*24.0*3600.0 # sec/day
        self.output_frequency = namelist['io']['output_frequency']*24.0*3600.0 # sec/day

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
        self.setup_stats_file(Pr, Gr)
        return

    cpdef open_files(self):
        self.root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        self.global_mean_grp = self.root_grp.groups['global_mean']
        self.axisymmetric_mean_grp = self.root_grp.groups['axisymmetric_mean']
        self.surface_axisymmetric_mean_grp = self.root_grp.groups['surface_axisymmetric_mean']
        return

    cpdef close_files(self):
        self.root_grp.close()
        return

    cpdef setup_stats_file(self, Parameters Pr, Grid Gr):
        cdef:
            Py_ssize_t imin = Gr.ng
            Py_ssize_t imax = Gr.nx+Gr.ng
            Py_ssize_t jmin = Gr.ng
            Py_ssize_t jmax = Gr.ny+Gr.ng

        root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')

        # Set coordinates
        coordinate_grp = root_grp.createGroup('coordinates')
        coordinate_grp.createDimension('x', Gr.nx)
        coordinate_grp.createDimension('y', Gr.ny)
        coordinate_grp.createDimension('r', Gr.nr)
        coordinate_grp.createDimension('lay', 1)

        x = coordinate_grp.createVariable('x', 'f8', ('x'))
        x[:] = Gr.x[imin:imax]
        y = coordinate_grp.createVariable('y', 'f8', ('y'))
        y[:] = Gr.y[jmin:jmax]
        r = coordinate_grp.createVariable('r', 'f8', ('r'))
        r[:] = Gr.r
        layer = coordinate_grp.createVariable('layers', 'f8',('lay'))
        layer[:] = np.array(Pr.n_layers)
        del x, y, layer

        # Set axisymmetric mean
        axisymmetric_mean_grp = root_grp.createGroup('axisymmetric_mean')
        axisymmetric_mean_grp.createDimension('time', None)
        axisymmetric_mean_grp.createDimension('r', Gr.nr)
        axisymmetric_mean_grp.createDimension('lay',  Pr.n_layers)
        t = axisymmetric_mean_grp.createVariable('t', 'f8',  ('time'))

        # Set global mean
        global_mean_grp = root_grp.createGroup('global_mean')
        global_mean_grp.createDimension('time', None)
        global_mean_grp.createDimension('lay',  Pr.n_layers)
        t = global_mean_grp.createVariable('t', 'f8', ('time'))

        # Set surface axisymmetric mean
        surface_axisymmetric_mean_grp = root_grp.createGroup('surface_axisymmetric_mean')
        surface_axisymmetric_mean_grp.createDimension('time', None)
        surface_axisymmetric_mean_grp.createDimension('r', Pr.nx)
        t = surface_axisymmetric_mean_grp.createVariable('t', 'f8', ('time'))
        root_grp.close()

        return

    cpdef add_global_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        global_grp = root_grp.groups['global_mean']
        new_var = global_grp.createVariable(var_name, 'f8',('time', 'lay'))
        root_grp.close()
        return

    cpdef add_axisymmetric_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        axisymmetric_mean_grp = root_grp.groups['axisymmetric_mean']
        new_var = axisymmetric_mean_grp.createVariable(var_name, 'f8', ('time','r','lay'))
        root_grp.close()
        return

    cpdef add_surface_axisymmetric_mean(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        surface_axisymmetric_mean_grp = root_grp.groups['surface_axisymmetric_mean']
        new_var = surface_axisymmetric_mean_grp.createVariable(var_name, 'f8', ('time','r'))
        root_grp.close()
        return


    cpdef write_global_mean(self, var_name, data):
        var = self.global_mean_grp.variables[var_name]
        var[-1,:] = np.array(np.mean(np.mean(data,0),0))
        return

    cpdef write_axisymmetric_mean(self, var_name, data):
        var = self.axisymmetric_mean_grp.variables[var_name]
        for k in range (self.nl):
            var[-1,:,k] = np.array(axisymmetric_mean(self.xc, self.yc, data.base[:,:,k]))
        return

    cpdef write_surface_axisymmetric_mean(self, var_name, data):
        var = self.surface_axisymmetric_mean_grp.variables[var_name]
        var[-1,:] = np.array(axisymmetric_mean(self.xc, self.yc, data.base))
        return

    cpdef write_3D_variable(self, Parameters Pr, Grid Gr, t, n_layers, var_name, data):
        cdef:
            Py_ssize_t imin = Gr.ng
            Py_ssize_t imax = Gr.nx+Gr.ng
            Py_ssize_t jmin = Gr.ng
            Py_ssize_t jmax = Gr.ny+Gr.ng

        root_grp = nc.Dataset(self.path_plus_var+var_name+'_'+str(t)+'.nc', 'w', format='NETCDF4')
        root_grp.createDimension('x', Gr.nx)
        root_grp.createDimension('y', Gr.ny)
        root_grp.createDimension('lay', Gr.nl)
        var = root_grp.createVariable(var_name, 'f8', ('x', 'y','lay'))
        var[:,:,:] = np.array(data[imin:imax,jmin:jmax,:])
        root_grp.close()
        return

    cpdef write_2D_variable(self, Parameters Pr, Grid Gr, t, var_name, data):
        cdef:
            Py_ssize_t imin = Gr.ng
            Py_ssize_t imax = Gr.nx+Gr.ng
            Py_ssize_t jmin = Gr.ng
            Py_ssize_t jmax = Gr.ny+Gr.ng

        root_grp = nc.Dataset(self.path_plus_var+var_name+'_'+str(t)+'.nc', 'w', format='NETCDF4')
        root_grp.createDimension('x', Pr.nx)
        root_grp.createDimension('y', Pr.ny)
        var = root_grp.createVariable(var_name, 'f8', ('x', 'y'))
        var[:,:] = np.array(data[imin:imax,jmin:jmax])
        root_grp.close()
        return

    cpdef write_simulation_time(self, t):
        # Write to global mean group
        global_mean_t = self.global_mean_grp.variables['t']
        global_mean_t[global_mean_t.shape[0]] = t

        # Write to axisymmetric mean group
        axisymmetric_mean_t = self.axisymmetric_mean_grp.variables['t']
        axisymmetric_mean_t[axisymmetric_mean_t.shape[0]] = t

        # Write to axisymmetric surface mean group
        surface_axisymmetric_mean_grp_t = self.surface_axisymmetric_mean_grp.variables['t']
        surface_axisymmetric_mean_grp_t[surface_axisymmetric_mean_grp_t.shape[0]] = t
        return