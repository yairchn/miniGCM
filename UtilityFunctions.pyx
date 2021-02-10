import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

# make sure that total moisture content is non-negative
cpdef set_min_vapour(qp,qbar):
    qtot = qp + qbar
    qtot[qtot<0] = 0
    return (qtot-qbar)

# Function for plotting KE spectra
cpdef keSpectra(Grid Gr, u, v):
    uk = Gr.grdtospec(u)
    vk = Gr.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(Gr.l)+1)
    k = np.arange(np.amax(Gr.l)+1)
    for i in range(0,np.amax(Gr.l)):
        Ek[i] = np.sum(Esp[np.logical_and(Gr.l>=i-0.5 , Gr.l<i+0.5)])
    return [Ek,k]