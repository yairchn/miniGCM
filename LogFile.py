import numpy as np
import os
# from DiagnosticVariables import DiagnosticVariables, DiagnosticVariable
# from PrognosticVariables import PrognosticVariables, PrognosticVariable
# from TimeStepping import TimeStepping
# from Parameters import Parameters

class LogFile:
    def __init__(self, namelist):
        return

    def initialize(self,  Pr, namelist):
        print('logfile ',Pr.logfilename)
        return

    def update(self, Pr, TS, DV, PV, wallclocktime):
        nl = Pr.n_layers

        os.system('echo elapsed time [days] about '
        	      +str(np.floor_divide(TS.t,(24.0*3600.0))) + '>> '+Pr.logfilename)
        os.system('echo wall-clock time [hours] about '
                  +str(np.floor_divide(wallclocktime,3600.0)) + '>> '+Pr.logfilename)
        os.system('echo efficiency [simtime / wallclock] '
                  +str(TS.t/wallclocktime) + '>> '+Pr.logfilename)
        os.system('echo estimated simulation time [hours] '
                  +str(TS.t_max/(TS.t/wallclocktime)/3600.0) + '>> '+Pr.logfilename)
        os.system('echo dt [sec]' + str(TS.dt) + '>> '+Pr.logfilename)

        os.system('echo p_surface min max ' + str(PV.P.values[:,:,nl].min())
                      + ' ' + str(PV.P.values[:,:,nl].max()) + '>> '+Pr.logfilename)

        for i in range(Pr.n_layers):
            os.system('echo u layer '+str(i+1)+' min max ' + str(DV.U.values[:,:,i].min())
            	      + ' ' + str(DV.U.values[:,:,i].max()) + '>> '+Pr.logfilename)

        for i in range(Pr.n_layers):
            os.system('echo T layer '+str(i+1)+' min max ' + str(PV.T.values[:,:,i].min())
                      + ' ' + str(PV.T.values[:,:,i].max()) + '>> '+Pr.logfilename)

        for i in range(Pr.n_layers):
            if Pr.moist_index > 0.0:
                os.system('echo QT layer '+str(i+1)+' min max ' + str(PV.QT.values[:,:,i].min())
                          + ' ' + str(PV.QT.values[:,:,i].max()) + '>> '+Pr.logfilename)
                os.system('echo dTdt_mp layer '+str(i+1)+' min max ' + str(PV.T.mp_tendency[:,:,i].min())
                          + ' ' + str(PV.T.mp_tendency[:,:,i].max()) + '>> '+Pr.logfilename)
        return