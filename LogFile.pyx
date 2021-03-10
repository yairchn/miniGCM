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

    cpdef update(self, Parameters Pr, TimeStepping TS, DiagnosticVariables DV, PrognosticVariables PV, wallclocktime):
        cdef:
            Py_ssize_t nl = Pr.n_layers

        os.system('echo elapsed time [days] about '
        	      +str(np.floor_divide(TS.t,(24.0*3600.0))) + '>> '+Pr.logfilename)
        os.system('echo wall-clock time [hours] about '
                  +str(np.floor_divide(wallclocktime,3600.0)) + '>> '+Pr.logfilename)
        os.system('echo efficiency [simtime / wallclock] '
                  +str(TS.t/wallclocktime) + '>> '+Pr.logfilename)
        os.system('echo estimated simulation length [hours] '
                  +str(TS.t_max/(TS.t/wallclocktime)/3600.0) + '>> '+Pr.logfilename)
        os.system('echo dt [sec]' + str(TS.dt) + '>> '+Pr.logfilename)

        os.system('echo p_s min max ' + str(PV.P.values.base[:,:,nl].min())
                + ' ' + str(PV.P.values.base[:,:,nl].max()) + '>> '+Pr.logfilename)
        for k in range(nl):
            os.system('echo u layer ' + str(k+1) + ' min max ' + str(DV.U.values.base[:,:,k].min())
                    + ' ' + str(DV.U.values.base[:,:,k].max()) + '>> '+Pr.logfilename)
            os.system('echo v layer ' + str(k+1) + ' min max ' + str(DV.V.values.base[:,:,k].min())
                    + ' ' + str(DV.V.values.base[:,:,k].max()) + '>> '+Pr.logfilename)
            os.system('echo T layer ' + str(k+1) + ' min max ' + str(PV.T.values.base[:,:,k].min())
                    + ' ' + str(PV.T.values.base[:,:,k].max()) + '>> '+Pr.logfilename)
            os.system('echo dTdt_mp layer ' + str(k+1) + ' min max ' + str(PV.T.mp_tendency.base[:,:,k].min())
                    + ' ' + str(PV.T.mp_tendency.base[:,:,k].max()) + '>> '+Pr.logfilename)
        return
