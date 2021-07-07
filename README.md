## miniGCM - A minimal Global Circulation Model in python

Author: Yair Cohen
Last updated : July 2021

INTALL

MAC OSX 
download + unzip + install:
1. fftw
2. miniconda
3. shtns version shtns-3.3.1-r694
4. xcode , xcode command tools

Create miniconda env with (python3.6.8 packages)
> conda create -n minigcm python=3.6.8 scipy numpy matplotlib cython netCDF4 xarray

cd to FFTW, 
> ./configure CC=gfortran --enable-openmp --enable-shared --prefix=/Users/yaircohen/opt/miniconda3/envs/minigcm
> make 
> make install

cd SHTNS, 
> ./configure --enable-openmp --enable-python --prefix=/Users/yaircohen/opt/miniconda3/envs/minigcm
> make 
> make install

check installation by
> python
> import shtns

clone miniGC
> git clone https://github.com/yairchn/miniGCM.git

activate env
> conda activate minigcm

generate namelist

> generate_namelist.py HeldSuarez

compile
> python setup.py build_ext --inplace 

run model
> python main.py HeldSuarez.in