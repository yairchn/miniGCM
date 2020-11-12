# from __future__ import print_function
import numpy as np
import time
from math import *
import os
import shutil
import netCDF4 as nc
import pylab as plt

cdef class Stats:
    cdef:
        object root_grp
        object variable_grp
        object meridional_mean_grp
        object zonal_mean_grp
        object global_mean_grp

        str stats_file_name
        str stats_path
        str output_path
        str path_plus_file
        str uuid
        public double last_output_time
        public double stats_frequency
        public double output_frequency

    cpdef open_files(self)
    cpdef close_files(self)
    cpdef setup_stats_file(self, Gr)
    cpdef add_global_mean(self, var_name)
    cpdef add_meridional_mean(self, var_name)
    cpdef add_zonal_mean(self, var_name)
    cpdef write_global_mean(self, var_name, data, t)
    cpdef write_meridional_mean(self, var_name, data, t)
    cpdef write_zonal_mean(self, var_name, data, t)
    cpdef write_3D_variable(self, Gr, t, n_layers, var_name, data)
    cpdef write_simulation_time(self, t)