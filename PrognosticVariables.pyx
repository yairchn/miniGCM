import cython
from Grid cimport Grid
from math import *
import matplotlib.pyplot as plt
import netCDF4
from NetCDFIO cimport NetCDFIO_Stats
import numpy as np
cimport numpy as np
import scipy as sc
import sys
from TimeStepping cimport TimeStepping

cdef class PrognosticVariable:
    def __init__(self, nx, ny, nl, n_spec, kind, name, units):
        self.values = np.zeros((nx,ny,nl),dtype=np.float64, order='c')
        self.spectral = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.old = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.now = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.tendency = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.VerticalFlux = np.zeros((nx,ny,nl+1),dtype=np.float64, order='c')
        self.sp_VerticalFlux = np.zeros((n_spec,nl),dtype = np.complex, order='c')
        self.forcing = np.zeros((n_spec,nl),dtype = np.complex, order='c')

        self.kind = kind
        self.name = name
        self.units = units
        return

cdef class PrognosticVariables:
    def __init__(self, Grid Gr):
        self.Vorticity   = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Vorticity' , 'zeta',  '1/s' )
        self.Divergence  = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Divergance', 'delta', '1/s' )
        self.T           = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Temperature'       ,  'T','K' )
        self.QT          = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers,  Gr.SphericalGrid.nlm,'Specific_humidity' ,  'qt','kg/kg' )
        self.P           = PrognosticVariable(Gr.nlats, Gr.nlons, Gr.n_layers+1,Gr.SphericalGrid.nlm,'Pressure'          ,  'p','pasc' )
        return

    cpdef initialize(self, Grid Gr):
        cdef:
            Py_ssize_t k
        self.T_init  = np.array([229.0, 257.0, 295.0])
        self.P_init  = np.array([Gr.p1, Gr.p2, Gr.p3, Gr.p_ref])
        self.QT_init = np.array([0.0, 0.0, 0.0])

        self.Vorticity.values  = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.float64, order='c')
        self.Divergence.values = np.zeros((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.float64, order='c')
        self.P.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers+1),  dtype=np.float64, order='c'),self.P_init)
        self.T.values          = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.float64, order='c'),self.T_init)
        self.QT.values         = np.multiply(np.ones((Gr.nlats, Gr.nlons, Gr.n_layers),  dtype=np.float64, order='c'),self.QT_init)
        # initilize spectral values
        for k in range(Gr.n_layers):
            self.P.spectral.base[:,k]           = Gr.SphericalGrid.grdtospec(self.Divergence.values.base[:,:,k])
            self.T.spectral.base[:,k]           = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,k])
            self.QT.spectral.base[:,k]          = Gr.SphericalGrid.grdtospec(self.T.values.base[:,:,k])
            self.Vorticity.spectral.base[:,k]   = Gr.SphericalGrid.grdtospec(self.QT.values.base[:,:,k])
            self.Divergence.spectral.base[:,k]  = Gr.SphericalGrid.grdtospec(self.Vorticity.values.base[:,:,k])
        self.P.spectral.base[:,Gr.n_layers] = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,Gr.n_layers])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_global_mean('global_mean_T')
        Stats.add_global_mean('global_mean_QT')
        Stats.add_zonal_mean('zonal_mean_P')
        Stats.add_zonal_mean('zonal_mean_T')
        Stats.add_zonal_mean('zonal_mean_QT')
        Stats.add_zonal_mean('zonal_mean_divergence')
        Stats.add_zonal_mean('zonal_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_divergence')
        Stats.add_meridional_mean('meridional_mean_vorticity')
        Stats.add_meridional_mean('meridional_mean_P')
        Stats.add_meridional_mean('meridional_mean_T')
        Stats.add_meridional_mean('meridional_mean_QT')
        return

    # convert spherical data to spectral
    # I needto define this function to ast on a general variable
    cpdef physical_to_spectral(self, Grid Gr):
        cdef:
            Py_ssize_t k
        for k in range(Gr.n_layers):
            self.Vorticity.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Vorticity.values.base[:,:,k])
            self.Divergence.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.Divergence.values.base[:,:,k])
            self.T.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.T.values.base[:,:,k])
            self.QT.spectral.base[:,k] = Gr.SphericalGrid.grdtospec(self.QT.values.base[:,:,k])
        self.P.spectral.base[:,Gr.n_layers] = Gr.SphericalGrid.grdtospec(self.P.values.base[:,:,Gr.n_layers])
        return

    cpdef spectral_to_physical(self, Grid Gr):
        cdef:
            Py_ssize_t k

        for k in range(Gr.n_layers):
            self.Vorticity.values.base[:,:,k]  = Gr.SphericalGrid.spectogrd(self.Vorticity.spectral.base[:,k])
            self.Divergence.values.base[:,:,k] = Gr.SphericalGrid.spectogrd(self.Divergence.spectral.base[:,k])
            self.T.values.base[:,:,k]          = Gr.SphericalGrid.spectogrd(self.T.spectral.base[:,k])
            self.QT.values.base[:,:,k]         = Gr.SphericalGrid.spectogrd(self.QT.spectral.base[:,k])
        self.P.values.base[:,:,Gr.n_layers] = Gr.SphericalGrid.spectogrd(np.copy(self.P.spectral.base[:,Gr.n_layers]))
        return

    # quick utility to set arrays with values in the "new" arrays
    cpdef set_old_with_now(self):
        self.Vorticity.old  = self.Vorticity.now.copy()
        self.Divergence.old = self.Divergence.now.copy()
        self.T.old          = self.T.now.copy()
        self.QT.old         = self.QT.now.copy()
        self.P.old          = self.P.now.copy()
        return

    cpdef set_now_with_tendencies(self):
        self.Vorticity.now  = self.Vorticity.tendency.copy()
        self.Divergence.now = self.Divergence.tendency.copy()
        self.T.now          = self.T.tendency.copy()
        self.QT.now         = self.QT.tendency.copy()
        self.P.now          = self.P.tendency.copy()
        return

    cpdef reset_pressures(self, Grid Gr):
        for k in range(Gr.n_layers):
            self.P.values.base[:,:,k] = np.add(np.zeros_like(self.P.values.base[:,:,k]),self.P_init[k])
        return

    # this should be done in time intervals and save each time new files,not part of stats 
    cpdef stats_io(self, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_global_mean('global_mean_T', self.T.values, TS.t)
        Stats.write_global_mean('global_mean_QT', self.QT.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_P',self.P.values[:,:,1:4], TS.t)
        Stats.write_zonal_mean('zonal_mean_T',self.T.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_QT',self.QT.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_divergence',self.Divergence.values, TS.t)
        Stats.write_zonal_mean('zonal_mean_vorticity',self.Vorticity.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_P',self.P.values[:,:,1:4], TS.t)
        Stats.write_meridional_mean('meridional_mean_T',self.T.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_QT',self.QT.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_divergence',self.Divergence.values, TS.t)
        Stats.write_meridional_mean('meridional_mean_vorticity',self.Vorticity.values, TS.t)
        return

    cpdef io(self, Grid Gr, TimeStepping TS, NetCDFIO_Stats Stats):
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Vorticity',         self.Vorticity.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Divergence',        self.Divergence.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Temperature',       self.T.values)
        Stats.write_3D_variable(Gr, int(TS.t),Gr.n_layers, 'Specific_humidity', self.QT.values)
        Stats.write_3D_variable(Gr, int(TS.t),1,           'Pressure',          self.P.values[:,:,Gr.n_layers])
        return

    cpdef compute_tendencies(self, Grid Gr, PrognosticVariables PV, DiagnosticVariables DV, namelist):
        cdef:
            Py_ssize_t k
        nz = Gr.n_layers
        # compute surface pressure tendency
        Vortical_P_flux, Divergent_P_flux = Gr.SphericalGrid.getvrtdivspec(
                         np.multiply(DV.U.values[:,:,nz-1],np.subtract(PV.P.values[:,:,nz-1],PV.P.values[:,:,nz])),
                         np.multiply(DV.V.values[:,:,nz-1],np.subtract(PV.P.values[:,:,nz-1],PV.P.values[:,:,nz]))) # Vortical_P_flux is not used
        PV.P.tendency[:,nz] = Divergent_P_flux + DV.Wp.spectral[:,nz-1]

        # compute vertical fluxes for vorticity, divergence, temperature and specific humity
        for k in range(nz):
            T_high  = PV.T.values[:,:,k]
            QT_high = PV.QT.values[:,:,k]
            U_high  = DV.U.values[:,:,k]
            V_high  = DV.V.values[:,:,k]
            if k==nz-1:
                T_low  = T_high
                QT_low = QT_high
                U_low  = U_high
                V_low  = V_high
            else:
                T_low  = PV.T.values[:,:,k+1]
                QT_low = PV.QT.values[:,:,k+1]
                U_low  = DV.U.values[:,:,k+1]
                V_low  = DV.V.values[:,:,k+1]
            # as a convention the vertical flux at k is comming ot the layer k from below (k+1)
            dp = np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])
            PV.T.VerticalFlux[:,:,k]  = np.divide(0.5*np.multiply(np.add(T_high,T_low),DV.Wp.values[:,:,k+1]),dp)
            PV.QT.VerticalFlux[:,:,k] = np.divide(0.5*np.multiply(np.add(QT_high,QT_low),DV.Wp.values[:,:,k+1]),dp)
            u_vertical_flux = np.multiply(np.multiply(0.5,DV.Wp.values[:,:,k+1]),np.divide(np.subtract(U_low,U_high),dp))
            v_vertical_flux = np.multiply(np.multiply(0.5,DV.Wp.values[:,:,k+1]),np.divide(np.subtract(V_low,V_high),dp))
            Vortical_momentum_flux, Divergent_momentum_flux = Gr.SphericalGrid.getvrtdivspec(u_vertical_flux, v_vertical_flux)
            PV.Vorticity.sp_VerticalFlux[:,k]  = Vortical_momentum_flux  # proportional to Wp[k+1] at the bottom of the k'th layer
            PV.Divergence.sp_VerticalFlux[:,k] = Divergent_momentum_flux # proportional to Wp[k+1] at the bottom of the k'th layer

        for k in range(nz):
            # dp_ratio is the ratio of dp between the layers used for correction of the vertical fluxes
            dp = np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])
            dp_m = np.subtract(PV.P.values[:,:,k],PV.P.values[:,:,k-1])
            if k==0:
                dp_ratio_ = np.multiply(0.0,PV.P.values[:,:,k])
            else:
                dp_ratio_ = np.divide(dp_m,dp)
            dp_ratio = Gr.SphericalGrid.grdtospec(dp_ratio_)
            # compute he energy laplacian for the divergence equation
            Dry_Energy_laplacian = Gr.SphericalGrid.lap*Gr.SphericalGrid.grdtospec(
                                   np.add(DV.gZ.values[:,:,k], DV.KE.values[:,:,k]))

            # compute vorticity diveregnce componenets of horizontal momentum fluxes and temperature fluxes  - new name
            Vorticity_flux, Divergence_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],np.add(PV.Vorticity.values[:,:,k],Gr.Coriolis)),
                                                  np.multiply(DV.V.values[:,:,k],np.add(PV.Vorticity.values[:,:,k],Gr.Coriolis)))
            Vortical_T_flux, Divergent_T_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.T.values[:,:,k]),
                                                  np.multiply(DV.V.values[:,:,k],PV.T.values[:,:,k])) # Vortical_T_flux is not used
            Vortical_QT_flux, Divergent_QT_flux = Gr.SphericalGrid.getvrtdivspec(np.multiply(DV.U.values[:,:,k],PV.QT.values[:,:,k]),
                                                  np.multiply(DV.V.values[:,:,k],PV.QT.values[:,:,k])) # Vortical_QT_flux is not used


            # compute thermal expension term for T equation
            dgZ = np.subtract(DV.gZ.values[:,:,k+1],DV.gZ.values[:,:,k])
            dp = np.subtract(PV.P.values[:,:,k+1],PV.P.values[:,:,k])
            Thermal_expension = Gr.SphericalGrid.grdtospec(np.multiply(np.divide(DV.Wp.values[:,:,k+1],Gr.cp),np.divide(dgZ/dp)))
            # compute tendencies
            PV.Divergence.tendency[:,k] = (Vorticity_flux - Dry_Energy_laplacian - PV.Divergence.sp_VerticalFlux[:,k]
                                        - np.multiply(PV.Divergence.sp_VerticalFlux[:,k-1],dp_ratio)
                                        + PV.Divergence.forcing[:,k])
            PV.Vorticity.tendency[:,k]  = (- Divergence_flux - PV.Vorticity.sp_VerticalFlux[:,k]
                                    - np.multiply(PV.Vorticity.sp_VerticalFlux[:,k-1],dp_ratio) - PV.Vorticity.forcing[:,k])
            PV.T.tendency[:,k] = (- Divergent_T_flux - Gr.SphericalGrid.grdtospec(PV.T.VerticalFlux[:,:,k] - np.multiply(PV.T.VerticalFlux[:,:,k-1],dp_ratio_))
                                  - Thermal_expension  + PV.T.forcing[:,k])
            # PV.QT.tendency[:,k] = - Divergent_QT_flux + Gr.SphericalGrid.grdtospec(-PV.QT.VerticalFlux[:,:,k] + np.multiply(PV.QT.VerticalFlux[:,:,k-1],dp_ratio_)) + PV.QT.forcing[:,k]
            PV.QT.tendency[:,k]          = np.zeros_like(PV.QT.spectral[:,k])

        return