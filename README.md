## miniGCM - A minimal Global Circulation Model in python

Authors: Yair Cohen and Josef Schröttle
Last updated : July 2021

miniGCM relays on SHTns to perform Spherical Harmonic Transforms.
For SHTns documentation and download see:
https://bitbucket.org/nschaeff/shtns/src/master/
https://www2.atmos.umd.edu/~dkleist/docs/shtns/doc/html/
Reference:
Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations
http://arxiv.org/abs/1202.6522

INTALL SHTns

We are providing here an example of installing SHTns using conda environment which we found relatively simple.
Other installation instructions can be found in the links above.

**MAC OSX**

download + unzip + install:
1. fftw
2. miniconda
3. shtns version shtns-3.3.1-r694
4. xcode , xcode command tools

Create miniconda env with (python3.6.8 packages)
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


clone miniGC
> git clone https://github.com/yairchn/miniGCM.git


generate namelist

> generate_namelist.py HeldSuarez

compile
> python setup.py build_ext --inplace 

run model
> python main.py HeldSuarez.in