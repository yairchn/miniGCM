## miniGCM - A minimal Global Circulation Model in python
Authors: Yair Cohen and Josef Schröttle
Last updated : July 2021

Mini GCM is a minimal General Circulation model code in a user friendly interface in python 
The model uses Cython/C kernels for faster computation and relays on SHTns to perform Spherical Harmonic Transforms.
For SHTns documentation and download see:
> https://bitbucket.org/nschaeff/shtns/src/master/

> https://www2.atmos.umd.edu/~dkleist/docs/shtns/doc/html/

For SHTns reference see:
Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations

> http://arxiv.org/abs/1202.6522

Before cloning miniGCM SHTns should be installed. 

We are providing here an example of installing SHTns using conda environment which we found relatively simple.
Other installation instructions can be found in the links above.

A - **Install SHTns**
Mac OSX:

download + unzip + install:
1. fftw (http://www.fftw.org/download.html)
2. miniconda (https://docs.conda.io/en/latest/miniconda.html)
3. shtns version shtns-3.3.1-r694 (https://bitbucket.org/nschaeff/shtns/downloads/)
4. xcode & xcode command tools (App store)

Create miniconda env with (python3.6.8 and relevant python packages)
> conda create -n minigcm python=3.6.8 scipy numpy matplotlib cython netCDF4 xarray

activate env
> conda activate minigcm

Keep the location of the environment as MY_PREFIX (for example)
> environment location: /Users/USERNAME/opt/miniconda3/envs/minigcm

cd to FFTW folder and
> ./configure CC=gfortran --enable-openmp --enable-shared --prefix=MY_PREFIX
> make 
> make install

cd SHTNS  folder and
> ./configure --enable-openmp --enable-python --prefix=MY_PREFIX
> make 
> make install

check installation by
> python
> import shtns

B - **clone miniGC**

> git clone https://github.com/yairchn/miniGCM.git

1. generate namelist

> generate_namelist.py HeldSuarez

2. compile

> python setup.py build_ext --inplace 

3. run model

> python main.py HeldSuarez.in



## notes for rrtmg - add openmpi-mpicc  with: conda install -c conda-forge openmpi-mpicc 