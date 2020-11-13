import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from NetCDFIO cimport NetCDFIO_Stats

class Microphysics:
    def __init__(self, nu, nz, loc, kind, name, units):
        return

    def initialize(self, GMV):
        return

    def initialize_io(self, NetCDFIO_Stats Stats):
        return

    def update(self, Gr):
        return

    def io(self, NetCDFIO_Stats Stats, Ref):
        Stats.write_variable('Rain', self.Rain.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        return
