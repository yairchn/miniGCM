import os
import numpy as np
import scipy as sc
from math import *

# Function for plotting KE spectra
def keSpectra(u,v):
    uk = x.grdtospec(u)
    vk = x.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
    return [Ek,k]


# def include(filename):
#     if os.path.exists(filename): 
#         execfile(filename)


# include('myfile.py')
