from Grid import Grid
import numpy as np
from PrognosticVariables import PrognosticVariables
from Parameters import Parameters

class Diffusion:
    def __init__(self):
        return

    def initialize(self, Pr, Gr, namelist):
        return

    def update(self, Pr, Gr, PV, double dt):
        nl = Pr.n_layers
        nlm = Gr.SphericalGrid.nlm
        laplacian  = Gr.SphericalGrid.lap

        with nogil:
            for i in range(nlm):
                diffusion_factor_meso_scale = (1.0/Pr.efold_meso*((laplacian[i]/laplacian[-1])**(Pr.dissipation_order/2.0)))
                diffusion_factor_grid_scale= (1.0/Pr.efold_grid*((laplacian[i]/laplacian[-1])**Pr.dissipation_order))
                HyperDiffusionFactor = np.exp(-dt*(diffusion_factor_meso_scale+diffusion_factor_grid_scale))
                for k in range(nl):
                    PV.P.spectral[i,k]          = HyperDiffusionFactor * PV.P.spectral[i,k]
                    PV.Vorticity.spectral[i,k]  = HyperDiffusionFactor * PV.Vorticity.spectral[i,k]
                    PV.Divergence.spectral[i,k] = HyperDiffusionFactor * PV.Divergence.spectral[i,k]
                    PV.T.spectral[i,k]          = HyperDiffusionFactor * PV.T.spectral[i,k]
                    PV.QT.spectral[i,k]         = HyperDiffusionFactor * PV.QT.spectral[i,k]
        return
