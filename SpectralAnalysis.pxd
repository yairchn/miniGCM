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
        public double flux_frequency
        public double spectral_frequency
        public double spinup_time
        public bint spectral_analysis
        double [:,:] KE_spectrum
        double [:,:] KE_Rot_spectrum
        double [:,:] KE_Div_spectrum
        double [:,:] int_KE_spec_flux_div
        double [:,:] KE_spec_flux_div

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist)
    cpdef compute_spectral_flux(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, TimeStepping TS)
    cpdef compute_turbulence_spectrum(self, Parameters Pr, Grid Gr, PrognosticVariables PV, TimeStepping TS)
    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats)