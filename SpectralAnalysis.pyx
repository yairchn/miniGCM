import sys
import cython
from concurrent.futures import ThreadPoolExecutor
import numpy as np
cimport numpy as np
from Parameters cimport Parameters
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport PrognosticVariables
from DiagnosticVariables cimport DiagnosticVariables
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats


cdef class SpectralAnalysis:

    def __init__(self, namelist):
        self.spectral_analysis = namelist['spectral_analysis']['sa_flag']
        return

    cpdef initialize(self, Parameters Pr, Grid Gr, namelist):
        self.flux_frequency = namelist['spectral_analysis']['flux_frequency']
        self.spectral_frequency = namelist['spectral_analysis']['spectral_frequency']
        self.spinup_time = namelist['spectral_analysis']['spinup_time']*24.0*3600.0 # to days
        # self.KE_spectrum = np.zeros((Gr.SphericalGrid.nlm,Pr.n_layers),dtype = np.complex, order='c')
        # self.KE_Rot_spectrum = np.zeros((Gr.SphericalGrid.nlm,Pr.n_layers),dtype = np.complex, order='c')
        # self.KE_Div_spectrum = np.zeros((Gr.SphericalGrid.nlm,Pr.n_layers),dtype = np.complex, order='c')
        self.KE_spectrum = np.zeros((np.amax(Gr.shtns_l)+1,Pr.n_layers), dtype = np.float64, order='c')
        self.KE_Rot_spectrum = np.zeros((np.amax(Gr.shtns_l)+1,Pr.n_layers), dtype = np.float64, order='c')
        self.KE_Div_spectrum = np.zeros((np.amax(Gr.shtns_l)+1,Pr.n_layers), dtype = np.float64, order='c')
        self.int_KE_spec_flux_div = np.zeros((np.amax(Gr.shtns_l)+1,Pr.n_layers), dtype = np.float64, order='c')
        self.KE_spec_flux_div = np.zeros((np.amax(Gr.shtns_l)+1,Pr.n_layers), dtype = np.float64, order='c')
        return
    #@cython.wraparound(False)
    # @cython.boundscheck(False)
    cpdef compute_spectral_flux(self, Parameters Pr, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV):
        cdef:
            Py_ssize_t k, i
            Py_ssize_t nl = Pr.n_layers
            double complex [:] u_spec, v_spec, E_spec,  uak, vak
            double [:,:] dp, dps_dx, dps_dy, dudx, dudy, dvdx, dvdy

        for k in range(nl):
            dp=np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])

            u_spec      = Gr.SphericalGrid.grdtospec(DV.U.values.base[:,:,k])
            v_spec      = Gr.SphericalGrid.grdtospec(DV.V.values.base[:,:,k])
            dudx, dudy  = Gr.SphericalGrid.getgrad(u_spec.base) # get gradients in grid space
            dvdx, dvdy  = Gr.SphericalGrid.getgrad(v_spec.base) # get gradients in grid space

            # Kinetic Energy flux due to advection and divergence (first two terms in Appendix B1)
            U_adv = np.add(np.multiply(DV.U.values[:,:,k],dudx), np.multiply(DV.V.values[:,:,k],dudy))
            V_adv = np.add(np.multiply(DV.U.values[:,:,k],dvdx), np.multiply(DV.V.values[:,:,k],dvdy))
            U_vert_adv = np.multiply(PV.Divergence.values[:,:,k], DV.U.values[:,:,k])*0.5
            V_vert_adv = np.multiply(PV.Divergence.values[:,:,k], DV.V.values[:,:,k])*0.5

            # only non-linear divergence and divergent component in flux
            uak = Gr.SphericalGrid.grdtospec(np.multiply(dp, U_adv + U_vert_adv))
            vak = Gr.SphericalGrid.grdtospec(np.multiply(dp, V_adv + V_vert_adv))

            E_spec = -np.add(np.multiply(uak, np.conj(u_spec)), np.multiply(vak, np.conj(v_spec)))

            if k==nl:
                dps_dx, dps_dy = Gr.SphericalGrid.getgrad(Gr.SphericalGrid.grdtospec(PV.P.values.base[:,:,k+1])) # get gradients in grid space
                # Kinetic Energy flux due to mass gradients (third term in Appendix B1)
                pressure_contribution = 0.5*Gr.SphericalGrid.grdtospec(np.add(np.multiply(DV.U.values[:,:,k],dps_dx),
                                                                              np.multiply(DV.V.values[:,:,k],dps_dy)))
                E_spec -= np.multiply(pressure_contribution,
                            np.add(np.multiply(u_spec, np.conj(u_spec)) ,np.multiply(v_spec, np.conj(v_spec))))

                # Kinetic Energy flux error due to mass flux inbalance in momentum equation (E2, equation 38)
                momentum_error_x = 0.5*Gr.SphericalGrid.grdtospec(np.multiply(DV.gZ.values.base[:,:,k],dps_dx))
                momentum_error_y = 0.5*Gr.SphericalGrid.grdtospec(np.multiply(DV.gZ.values.base[:,:,k],dps_dy))
                E_spec += momentum_error_x*np.conj(u_spec) + momentum_error_y*np.conj(v_spec)

            # descritization error
            if k>0:
                uak = Gr.SphericalGrid.grdtospec(np.multiply(DV.Wp.values[:,:,k], DV.U.values[:,:,k]))
                vak = Gr.SphericalGrid.grdtospec(np.multiply(DV.Wp.values[:,:,k], DV.V.values[:,:,k]))
                uakD = Gr.SphericalGrid.grdtospec(np.multiply(DV.Wp.values[:,:,k], DV.U.values[:,:,k-1]))
                vakD = Gr.SphericalGrid.grdtospec(np.multiply(DV.Wp.values[:,:,k], DV.V.values[:,:,k-1]))
                u_specD = Gr.SphericalGrid.grdtospec(DV.U.values.base[:,:,k-1])
                v_specD = Gr.SphericalGrid.grdtospec(DV.V.values.base[:,:,k-1])

                usp  = 0.5*np.multiply(uak, np.conj(u_spec))
                vsp  = 0.5*np.multiply(vak, np.conj(v_spec))
                uspD = 0.5*np.multiply(uakD, np.conj(u_specD))
                vspD = 0.5*np.multiply(vakD, np.conj(v_specD))
                E_spec -= (uspD - usp + vspD - vsp)

            # boundary condition error
            if k==nl:
                uak = Gr.SphericalGrid.grdtospec(np.multiply(DV.Wp.values[:,:,k+1], DV.U.values[:,:,k]))
                vak = Gr.SphericalGrid.grdtospec(np.multiply(DV.Wp.values[:,:,k+1], DV.V.values[:,:,k]))

                usp  = 0.5*np.multiply(uak, np.conj(u_spec))
                vsp  = 0.5*np.multiply(vak, np.conj(v_spec))
                E_spec -= (usp + vsp)

            # Selection of wavenumbers
            for i in range(0,np.amax(Gr.shtns_l)+1):
                self.KE_spec_flux_div[i,k] = np.sum(E_spec.base[np.logical_and(Gr.shtns_l>=np.double(i-0.5), Gr.shtns_l<np.double(i+0.5))])
            # running sum (Integral)
            for i in range(0,np.amax(Gr.shtns_l)+1):
                for j in range(i,np.amax(Gr.shtns_l)+1):
                    self.int_KE_spec_flux_div[i,k] += self.KE_spec_flux_div[j,k]
            # for i in range(0,np.amax(l)+1):
            #     self.int_KE_spec_flux_div[i,k] = np.sum(self.KE_spec_flux_div[i:,k])
        return

    cpdef compute_turbulence_spectrum(self, Parameters Pr, Grid Gr, PrognosticVariables PV):
        cdef:
            # Py_ssize_t i,j,k
            Py_ssize_t nl = Pr.n_layers
            Py_ssize_t nlm = Gr.SphericalGrid.nlm

        factor = np.divide(np.divide(Pr.rsphere**2,Gr.shtns_l),np.add(Gr.shtns_l,1))

        for k in range(nl):
            Vort2_spect = np.multiply(factor, np.multiply(PV.Vorticity.spectral.base[:,k], np.conj(PV.Vorticity.spectral.base[:,k])))
            Div2_spect = np.multiply(factor, np.multiply(PV.Divergence.spectral.base[:,k], np.conj(PV.Divergence.spectral.base[:,k])))
            KE_spectrum = np.add(Vort2_spect, Div2_spect)
            for i in range(0,np.amax(Gr.shtns_l)):
                self.KE_spectrum[i,k] += np.sum(np.real(KE_spectrum[np.double(Gr.shtns_l)==i]), axis = 0)
                self.KE_Rot_spectrum[i,k] += np.sum(np.real(Vort2_spect.real[np.double(Gr.shtns_l)==i]), axis = 0)
                self.KE_Div_spectrum[i,k] += np.sum(np.real(Div2_spect.real[np.double(Gr.shtns_l)==i]), axis = 0)
        return

    cpdef io(self, Parameters Pr, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_spectral_analysis(Gr.wavenumbers.len(), Pr.n_layers, 'KE_spectrum', self.KE_spectrum)
        Stats.write_spectral_analysis(Gr.wavenumbers.len(), Pr.n_layers, 'KE_Rot_spectrum', self.KE_Rot_spectrum)
        Stats.write_spectral_analysis(Gr.wavenumbers.len(), Pr.n_layers, 'KE_Div_spectrum', self.KE_Div_spectrum)
        Stats.write_spectral_analysis(Gr.wavenumbers.len(), Pr.n_layers, 'KE_spec_flux_div', self.KE_spec_flux_div)
        Stats.write_spectral_analysis(Gr.wavenumbers.len(), Pr.n_layers, 'int_KE_spec_flux_div', self.int_KE_spec_flux_div)
        return
