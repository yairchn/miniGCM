import numpy as np
import scipy as sc
from math import *

class SurfaceBase:
    def __init__(self):
        return
    def initialize(self, Pr, Gr, PV):
        return
    def update(self, Pr, Gr, TS, PV):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class SurfaceNone(SurfaceBase):
    def __init__(self):
        return
    def initialize(self, Pr, Gr, PV):
        return
    def update(self, Pr, Gr, TS, PV):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return

class Surface_BulkFormula(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    def initialize(self, Pr, Gr, PV):
        return
    def update(self, Pr, Gr, TS, PV):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Pr, TS, Stats):
        return
