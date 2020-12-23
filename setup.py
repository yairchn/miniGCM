from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import sys
import platform
import subprocess as sp
import os.path
import string

# Now get include paths from relevant python modules
# include_path = [mpi4py.get_include()]

include_path = [np.get_include()]
# include_path += ['./Csrc']

if sys.platform == 'darwin':
    #Compile flags for MacOSX
    library_dirs = []
    libraries = []
    extensions = []
    extra_compile_args = []
    extra_compile_args += ['-O3', '-march=native', '-Wno-unused', '-Wno-#warnings','-fPIC']
    # extra_objects=['./RRTMG/rrtmg_build/rrtmg_combined.o']
    extra_objects = []
    netcdf_include = '/opt/local/include'
    netcdf_lib = '/opt/local/lib'
    f_compiler = 'gfortran'
elif 'linux' in sys.platform:
    #Compile flags for Travis (linux)
    library_dirs = []
    libraries = []
    #libraries.append('mpi')
    #libraries.append('gfortran')
    extensions = []
    extra_compile_args  = []
    extra_compile_args += ['-std=c99', '-O3', '-march=native', '-Wno-unused',
                           '-Wno-#warnings', '-Wno-maybe-uninitialized', '-Wno-cpp', '-Wno-array-bounds','-fPIC']
    from distutils.sysconfig import get_python_lib
    tmp_path = get_python_lib()
    netcdf_include = tmp_path + '/netcdf4/include'
    netcdf_lib = tmp_path + "/netcdf4/lib"
    f_compiler = 'gfortran'
else:
    print('Unknown system platform: ' + sys.platform  + 'or unknown system name: ' + platform.node())
    sys.exit()


_ext = Extension('PrognosticVariables', ['PrognosticVariables.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('DiagnosticVariables', ['DiagnosticVariables.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Forcing', ['Forcing.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('UtilityFunctions', ['UtilityFunctions.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)


_ext = Extension('Grid', ['Grid.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)


_ext = Extension('Simulation', ['Simulation.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)


_ext = Extension('Diffusion', ['Diffusion.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

# _ext = Extension('Microphysics', ['Microphysics.pyx'], include_dirs=include_path,
#                  extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
#                  runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('NetCDFIO', ['NetCDFIO.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('sphericalForcing', ['sphericalForcing.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Thermodynamics', ['Thermodynamics.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)


_ext = Extension('Surface', ['Surface.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

_ext = Extension('Cases', ['Cases.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

# _ext = Extension('sphTrans', ['sphTrans.pyx'], include_dirs=include_path,
#                  extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
#                  runtime_library_dirs=library_dirs)
# extensions.append(_ext)

_ext = Extension('TimeStepping', ['TimeStepping.pyx'], include_dirs=include_path,
                 extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
                 runtime_library_dirs=library_dirs)
extensions.append(_ext)

# _ext = Extension('pytest_wrapper', ['pytest_wrapper.pyx'], include_dirs=include_path,
#                  extra_compile_args=extra_compile_args, libraries=libraries, library_dirs=library_dirs,
#                  runtime_library_dirs=library_dirs)
# extensions.append(_ext)

setup(
    ext_modules=cythonize(extensions, verbose=1, include_path=include_path)
)
