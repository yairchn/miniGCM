import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from PrognosticVariables cimport PrognosticVariables
from Parameters cimport Parameters
from libc.math cimport exp

# overload exp for complex number
cdef extern from "complex.h" nogil:
    double complex exp(double complex z)

cdef class Diffusion:

    def __init__(self):
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        self.dxi_2 = 1.0/(Gr.dx*Gr.dx)
        self.dyi_2 = 1.0/(Gr.dy*Gr.dy)
        self.dxi_4 = 1.0/(Gr.dx*Gr.dx*Gr.dx*Gr.dx)
        self.dyi_4 = 1.0/(Gr.dy*Gr.dy*Gr.dy*Gr.dy)
        return
    #@cython.wraparound(False)
    # @cython.boundscheck(False)
    # cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
    #     cdef:
    #         Py_ssize_t i,j,k
    #         Py_ssize_t ng = Gr.ng
    #         Py_ssize_t nx = Gr.nx
    #         Py_ssize_t ny = Gr.ny
    #         Py_ssize_t nl = Gr.nl
    #         double HyperDiffusionFactor = -(1.0/Pr.efold)

        
    #     # with nogil:
    #     for i in range(ng,nx+1+ng):
    #         for j in range(ng,ny+ng):
    #             for k in range(nl):
    #                 PV.U.HyperDiffusion[i,j,k] = HyperDiffusionFactor * ((PV.U.values[i+2,j,k]
    #                                                                  -4.0*PV.U.values[i+1,j,k]
    #                                                                  +6.0*PV.U.values[i,j,k]
    #                                                                  -4.0*PV.U.values[i-1,j,k]
    #                                                                      +PV.U.values[i-2,j,k])*self.dxi_4
    #                                                                     +(PV.U.values[i,j+2,k]
    #                                                                  -4.0*PV.U.values[i,j+1,k]
    #                                                                  +6.0*PV.U.values[i,j,k]
    #                                                                  -4.0*PV.U.values[i,j-1,k]
    #                                                                      +PV.U.values[i,j-2,k])*self.dyi_4)
    #     for i in range(ng,nx+ng):
    #         for j in range(ng,ny+1+ng):
    #             for k in range(nl):
    #                 PV.V.HyperDiffusion[i,j,k] = HyperDiffusionFactor * ((PV.V.values[i+2,j,k]
    #                                                                  -4.0*PV.V.values[i+1,j,k]
    #                                                                  +6.0*PV.V.values[i,j,k]
    #                                                                  -4.0*PV.V.values[i-1,j,k]
    #                                                                      +PV.V.values[i-2,j,k])*self.dxi_4
    #                                                                     +(PV.V.values[i,j+2,k]
    #                                                                  -4.0*PV.V.values[i,j+1,k]
    #                                                                  +6.0*PV.V.values[i,j,k]
    #                                                                  -4.0*PV.V.values[i,j-1,k]
    #                                                                      +PV.V.values[i,j-2,k])*self.dyi_4)
    #     for i in range(ng,nx+ng):
    #         for j in range(ng,ny+ng):
    #             for k in range(nl):
    #                 PV.H.HyperDiffusion[i,j,k] = HyperDiffusionFactor * ((PV.H.values[i+2,j,k]
    #                                                                  -4.0*PV.H.values[i+1,j,k]
    #                                                                  +6.0*PV.H.values[i,j,k]
    #                                                                  -4.0*PV.H.values[i-1,j,k]
    #                                                                      +PV.H.values[i-2,j,k])*self.dxi_4
    #                                                                     +(PV.H.values[i,j+2,k]
    #                                                                  -4.0*PV.H.values[i,j+1,k]
    #                                                                  +6.0*PV.H.values[i,j,k]
    #                                                                  -4.0*PV.H.values[i,j-1,k]
    #                                                                      +PV.H.values[i,j-2,k])*self.dyi_4)
    #                 PV.QT.HyperDiffusion[i,j,k] = HyperDiffusionFactor * ((PV.QT.values[i+2,j,k]
    #                                                                   -4.0*PV.QT.values[i+1,j,k]
    #                                                                   +6.0*PV.QT.values[i,j,k]
    #                                                                   -4.0*PV.QT.values[i-1,j,k]
    #                                                                       +PV.QT.values[i-2,j,k])*self.dxi_4
    #                                                                      +(PV.QT.values[i,j+2,k]
    #                                                                   -4.0*PV.QT.values[i,j+1,k]
    #                                                                   +6.0*PV.QT.values[i,j,k]
    #                                                                   -4.0*PV.QT.values[i,j-1,k]
    #                                                                       +PV.QT.values[i,j-2,k])*self.dyi_4)
    #     return
    @cython.boundscheck(False)
    cpdef update(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t ng = Gr.ng
            Py_ssize_t nx = Gr.nx
            Py_ssize_t ny = Gr.ny
            Py_ssize_t nl = Gr.nl
            double HyperDiffusionFactor = -(1.0/Pr.efold)

        # with nogil:
        for i in range(ng,nx+ng):
            for j in range(ng,ny+ng):
                for k in range(nl):
                    PV.U.HyperDiffusion[i,j,k] = HyperDiffusionFactor * ((PV.U.values[i+1,j,k]
                                                                     -2.0*PV.U.values[i,j,k]
                                                                         +PV.U.values[i-1,j,k])*self.dxi_2
                                                                        +(PV.U.values[i,j+1,k]
                                                                     -2.0*PV.U.values[i,j,k]
                                                                         +PV.U.values[i,j-1,k])*self.dyi_2)
                    PV.H.HyperDiffusion[i,j,k] = HyperDiffusionFactor * ((PV.H.values[i+1,j,k]
                                                                     -2.0*PV.H.values[i,j,k]
                                                                         +PV.H.values[i-1,j,k])*self.dxi_2
                                                                        +(PV.H.values[i,j+1,k]
                                                                     -2.0*PV.H.values[i,j,k]
                                                                         +PV.H.values[i,j-1,k])*self.dyi_2)
                    PV.QT.HyperDiffusion[i,j,k] = HyperDiffusionFactor * ((PV.QT.values[i+1,j,k]
                                                                      -2.0*PV.QT.values[i,j,k]
                                                                          +PV.QT.values[i-1,j,k])*self.dxi_2
                                                                         +(PV.QT.values[i,j+1,k]
                                                                      -2.0*PV.QT.values[i,j,k]
                                                                          +PV.QT.values[i,j-1,k])*self.dyi_2)
        return
