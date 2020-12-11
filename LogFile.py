import numpy as np
from math import *

class LogFile:
    def __init__(self, namelist):
        return

    def initialize(self, namelist):
        self.uuid = namelist['meta']['uuid']
        self.casename = namelist['meta']['casename']
        self.outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                           + self.uuid[len(self.uuid )-5:len(self.uuid)]))
        self.logfile = self.outpath+'/'+self.casename+'.log'
        print('logfile ',self.logfile)
        return

    def update(self, TS, DV, namelist):
    	#write to stdoutput
        print('elapsed time [days] about', np.floor_divide(self.TS.t,(24.0*3600.0)))
        #write logfile
        os.system('echo elapsed time [days] about '
        	      +str(np.floor_divide(self.TS.t,(24.0*3600.0))) + '>> '+self.logfile)
        os.system('echo u layer 1 min max ' + str(self.DV.U.values[:,:,0].min())
        	      + ' ' + str(self.DV.U.values[:,:,0].max()) + '>> '+self.logfile)
        os.system('echo u layer 2 min max ' + str(self.DV.U.values[:,:,1].min())
        	      + ' ' + str(self.DV.U.values[:,:,1].max()) + '>> '+self.logfile)
        os.system('echo u layer 3 min max ' + str(self.DV.U.values[:,:,2].min())
        	      + ' ' + str(self.DV.U.values[:,:,2].max()) + '>> '+self.logfile)
        return