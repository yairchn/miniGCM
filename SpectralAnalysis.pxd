import sys
import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats


cdef class SpectralAnalysis:
    cdef:
        double flux_frequency
        double spectral_frequency
        double spinup_time
        bint spectral_analysis
        double [:,:] KE_spectrum
        double [:,:] KE_Rot_spectrum
        double [:,:] KE_Div_spectrum
    cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
    cpdef compute_spectral_flux(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV)
    cpdef compute_turbulence_spectrum(self, Parameters Pr, Grid Gr, PrognosticVariables PV)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)