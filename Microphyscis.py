import matplotlib.pyplot as plt
import numpy as np
from math import *

class Microphysics:
    def __init__(self, nu, nz, loc, kind, name, units):
        return

    def initialize(self, GMV):
        return

    def initialize_io(self, Stats):
        return

    def update(self, Gr):
        return

    def io(self, Stats, Ref):
        Stats.write_variable('Rain', self.Rain.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        return
