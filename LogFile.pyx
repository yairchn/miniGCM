import numpy as np
import os
from DiagnosticVariables cimport DiagnosticVariables, DiagnosticVariable
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

cdef class LogFile:
    def __init__(self, namelist):
        return

    cpdef initialize(self,  Parameters Pr, namelist):
        print('logfile ',Pr.logfilename)
        return

    cpdef update(self, Parameters Pr, TimeStepping TS, DiagnosticVariables DV, PrognosticVariables PV, wallclocktime, namelist):
        os.system('echo elapsed time [days] about '
        	      +str(np.floor_divide(TS.t,(24.0*3600.0))) + '>> '+Pr.logfilename)
        os.system('echo wall-clock time [hours] about '
                  +str(np.floor_divide(wallclocktime,3600.0)) + '>> '+Pr.logfilename)
        os.system('echo u layer 1 min max ' + str(DV.U.values.base[:,:,0].min())
        	      + ' ' + str(DV.U.values.base[:,:,0].max()) + '>> '+Pr.logfilename)
        os.system('echo u layer 2 min max ' + str(DV.U.values.base[:,:,1].min())
        	      + ' ' + str(DV.U.values.base[:,:,1].max()) + '>> '+Pr.logfilename)
        os.system('echo u layer 3 min max ' + str(DV.U.values.base[:,:,2].min())
        	      + ' ' + str(DV.U.values.base[:,:,2].max()) + '>> '+Pr.logfilename)
        return