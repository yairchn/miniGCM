import numpy as np
import scipy as sc
from math import *

class SurfaceBase:
    def __init__(self):
        return
    def initialize(self, Pr, Gr, namelist):
        return
    def update(self, Pr, Gr, PV, TS):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Stats):
        return

class SurfaceNone(SurfaceBase):
    def __init__(self):
        return
    def initialize_surface(self, Pr, Gr):
        return
    def update_surface(self, Pr, Gr):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Stats):
        return

class Surface_BulkFormula(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    def initialize_surface(self, Pr, Gr):
        return
    def update_surface(self, Pr, Gr):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Stats):
        return
