import numpy as np
import os

  class LogFile:
    def __init__(self, namelist):
        return

    def initialize(self, namelist):
        self.uuid = namelist['meta']['uuid']
        self.casename = namelist['meta']['casename']
        self.outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                            + self.uuid[len(self.uuid )-5:len(self.uuid)]))
        self.filename = self.outpath+'/'+self.casename+'.log'
        print('logfile ',self.filename)
        return

    def update(self, TS, DV, wallclocktime, namelist):
        #write logfile
        os.system('echo elapsed time [days] about '
        	      +str(np.floor_divide(TS.t,(24.0*3600.0))) + '>> '+self.filename)
        os.system('wall-clock time [hours] about '
                  +str(np.floor_divide(wallclocktime,3600.0)) + '>> '+self.filename)
        os.system('echo u layer 1 min max ' + str(DV.U.values[:,:,0].min())
        	      + ' ' + str(DV.U.values[:,:,0].max()) + '>> '+self.filename)
        os.system('echo u layer 2 min max ' + str(DV.U.values[:,:,1].min())
        	      + ' ' + str(DV.U.values[:,:,1].max()) + '>> '+self.filename)
        os.system('echo u layer 3 min max ' + str(DV.U.values[:,:,2].min())
        	      + ' ' + str(DV.U.values[:,:,2].max()) + '>> '+self.filename)
        return