import numpy as np
import os
from DiagnosticVariables cimport DiagnosticVariables, DiagnosticVariable
from PrognosticVariables cimport PrognosticVariables, PrognosticVariable
from TimeStepping cimport TimeStepping
from Parameters cimport Parameters

def class LogFile:
    def __init__(self, namelist):
        return

    def initialize(self,  Parameters Pr, namelist):
        print('logfile ',Pr.logfilename)
        return

    def update(self, Parameters Pr, TimeStepping TS, DiagnosticVariables DV, PrognosticVariables PV, wallclocktime):
        def:
            Py_ssize_t i
            Py_ssize_t nl = Pr.n_layers

        os.system('echo elapsed time [days] about '
        	      +str(np.floor_divide(TS.t,(24.0*3600.0))) + '>> '+Pr.logfilename)
        os.system('echo wall-clock time [hours] about '
                  +str(np.floor_divide(wallclocktime,3600.0)) + '>> '+Pr.logfilename)
        os.system('echo efficiency [simtime / wallclock] '
                  +str(TS.t/wallclocktime) + '>> '+Pr.logfilename)
        os.system('echo estimated simulation time [hours] '
                  +str(TS.t_max/(TS.t/wallclocktime)/3600.0) + '>> '+Pr.logfilename)
        os.system('echo dt [sec]' + str(TS.dt) + '>> '+Pr.logfilename)

        os.system('echo p_surface min max ' + str(PV.P.values.base[:,:,nl].min())
                      + ' ' + str(PV.P.values.base[:,:,nl].max()) + '>> '+Pr.logfilename)

        for i in range(Pr.n_layers):
            os.system('echo u layer '+str(i+1)+' min max ' + str(DV.U.values.base[:,:,i].min())
            	      + ' ' + str(DV.U.values.base[:,:,i].max()) + '>> '+Pr.logfilename)

        for i in range(Pr.n_layers):
            os.system('echo T layer '+str(i+1)+' min max ' + str(PV.T.values.base[:,:,i].min())
                      + ' ' + str(PV.T.values.base[:,:,i].max()) + '>> '+Pr.logfilename)

        for i in range(Pr.n_layers):
            if Pr.moist_index > 0.0:
                os.system('echo QT layer '+str(i+1)+' min max ' + str(PV.QT.values.base[:,:,i].min())
                          + ' ' + str(PV.QT.values.base[:,:,i].max()) + '>> '+Pr.logfilename)
                os.system('echo dTdt_mp layer '+str(i+1)+' min max ' + str(PV.T.mp_tendency.base[:,:,i].min())
                          + ' ' + str(PV.T.mp_tendency.base[:,:,i].max()) + '>> '+Pr.logfilename)
        return