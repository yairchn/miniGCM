import numpy as np
import os
from DiagnosticVariables cimport DiagnosticVariables, DiagnosticVariable
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

cdef class LogFile:
    cdef:
        str uuid
        str casename
        str outpath
        str filename

    cpdef initialize(self,  Parameters Pr, namelist)
    cpdef update(self, Parameters Pr, TimeStepping TS, DiagnosticVariables DV, PrognosticVariables PV, wallclocktime)