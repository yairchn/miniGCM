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
    cdef:
        object root_grp
        object variable_grp
        object meridional_mean_grp
        object zonal_mean_grp
        object global_mean_grp
        object surface_zonal_mean_grp

        str stats_file_name
        str stats_path
        str output_path
        str path_plus_file
        str path_plus_var
        str uuid
        public double last_output_time
        public double stats_frequency
        public double output_frequency
        double xc
        double yc

    cpdef open_files(self)
    cpdef close_files(self)
    cpdef setup_stats_file(self, Parameters Pr, Grid Gr)
    cpdef add_global_mean(self, var_name)
    cpdef add_axisymmetric_mean(self, var_name)
    cpdef add_surface_axisymmetric_mean(self, var_name)
    cpdef write_global_mean(self, var_name, data)
    cpdef write_axisymmetric_mean(self, var_name, data)
    cpdef write_surface_axisymmetric_mean(self, var_name, data)
    cpdef write_3D_variable(self, Parameters Pr, Grid Gr, t, n_layers, var_name, data)
    cpdef write_2D_variable(self, Parameters Pr, Grid Gr, t, var_name, data)
    cpdef write_simulation_time(self, t)