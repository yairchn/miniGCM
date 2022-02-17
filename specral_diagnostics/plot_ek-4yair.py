#Code to solve moist shallow water equations on a sphere
#adopted from SWE of the vorticity-divergence form.
#
# plotting script to calculate
# ============================
#  zonal means (iplot_means) 
#  contours (iplot_ctr)
#  spectra (iplot_spctr)
#  spectra (iplot_timsr)
#
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import netCDF4
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

hres_scaling=16
hres_scaling=1
hres_scaling=4

nlons  = hres_scaling*128  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)   # number of lats for gaussian grid.

# parameters for test
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5  # rotation rate
grav    = 9.80616   # gravity
Hbar    = 300.      # mean height (some typical range)


def energy(u,v,l,rsphere= rsphere):
    print('u.shape ',u.shape)
    print('v.shape ',v.shape)
    vrtsp,divsp = x.getvrtdivspec(u,v)
    factor = (rsphere**2)/l/(l+1)
    TotEsp = factor*(vrtsp*vrtsp.conj()+divsp*divsp.conj()).real
    RotEsp = factor*(vrtsp*vrtsp.conj()).real
    DivEsp = factor*(divsp*divsp.conj()).real
    TotEk = np.zeros(np.amax(l)+1)
    RotEk = np.zeros(np.amax(l)+1)
    DivEk = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)
    for i in range(0,np.amax(l)):
        TotEk[i] = np.sum(TotEsp[l==i])
        RotEk[i] = np.sum(RotEsp[l==i])
        DivEk[i] = np.sum(DivEsp[l==i])
    return [TotEk,RotEk,DivEk,k]


def Enstrophy_flux(u,v):
    vrtsp,divsp = x.getvrtdivspec(u,v)

    div = x.spectogrd(divsp)
    vrt = x.spectogrd(vrtsp)
    vrt_x, vrt_y = x.getgrad(vrtsp) # get gradients in grid space

    vrtsp_advection = x.grdtospec(vrt_x*u + vrt_y*v)

    Enstrophy_flux = -1.*vrtsp*vrtsp_advection.conj() - 1.*vrtsp.conj()*vrtsp_advection

    # build spectrum
    Enstrophy_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)

    # running sum (Integral)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Enstrophy_flux[np.logical_and(l>=i-0.5 , l<i+0.5)])
    for i in range(0,np.amax(l)):
        for j in range(i,np.amax(l)):
            Enstrophy_sum[i]+=Ek[j]

    return [Enstrophy_sum,k]



def keSpectral_flux_error_dp(u,v,Geo,pU,pD,uD,vD,Wp):
    uk  = x.grdtospec(u)
    vk  = x.grdtospec(v)
    ukD = x.grdtospec(uD)
    vkD = x.grdtospec(vD)
    uak = x.grdtospec(Wp*u)
    vak = x.grdtospec(Wp*v)
    uakD= x.grdtospec(Wp*uD)
    vakD= x.grdtospec(Wp*vD)

    usp = 0.5*uak*uk.conj()
    vsp = 0.5*vak*vk.conj()
    uspD= 0.5*uakD*ukD.conj()
    vspD= 0.5*vakD*vkD.conj()
    Esp = uspD - usp + vspD - vsp

    # build spectrum, initialize arrays
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    ks = np.arange(np.amax(l)+1)

    # Selection of wavenumbers
    for i in range(0,np.amax(l)+1):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
    # running sum (Integral)
    for i in range(0,np.amax(l)+1):
        for j in range(i,np.amax(l)+1):
            Ek_sum[i]+=Ek[j]

    return [Ek_sum,ks]


def keSpectral_flux_dp(u,v,Geo,pU,pD):
    zero=np.copy(u)*0.

    # get vorticity, divergence in spectral space 
    vrtsp,divsp = x.getvrtdivspec(u,v)
    vrt=x.spectogrd(vrtsp)
    div=x.spectogrd(divsp)

    # pressure Up is pi+1
    # pressure Down pi-1
    # for pressure weighted ke flux
    dp=pU-pD

    uk      = x.grdtospec(u)
    vk      = x.grdtospec(v)
    dpk     = x.grdtospec(dp)
    dpx,dpy = x.getgrad(dpk) # get gradients in grid space
    psx,psy = x.getgrad(x.grdtospec(pU)) # get gradients in grid space
    ux, uy  = x.getgrad(uk) # get gradients in grid space
    vx, vy  = x.getgrad(vk) # get gradients in grid space
    Geok = x.grdtospec(Geo)
    Geox, Geoy = x.getgrad(Geok) # get gradients in grid space
    # Kinetic Energy flux due to advection and divergence (first two terms in Appendix B1) 
    uak = x.grdtospec(dp*ux*u + dp*uy*v + dp*div*u/2.)  # only non-linear divergence and divergent component in flux
    vak = x.grdtospec(dp*vx*u + dp*vy*v + dp*div*v/2.)

    Usp = -1.*uak*uk.conj()
    Vsp = -1.*vak*vk.conj()
    Esp = Usp + Vsp 

    # Kinetic Energy flux due to mass gradients (third term in Appendix B1) 
    pressure_contribution = x.grdtospec(u*dpx/2. + v*dpy/2.) 
    Esp-= pressure_contribution*uk*uk.conj() + pressure_contribution*vk*vk.conj() 

    # Kinetic Energy flux error due to mass flux inbalance in momentum equation (E2, equation 38) 
    momentum_error_x = x.grdtospec(0.5*Geo*psx)
    momentum_error_y = x.grdtospec(0.5*Geo*psy)
    Esp+= momentum_error_x*uk.conj() + momentum_error_y*vk.conj()

    # build spectrum, initialize arrays
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    ks = np.arange(np.amax(l)+1)

    # select wavenumbers 
    for i in range(0,np.amax(l)+1):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])

    # running sum (Integral)
    for i in range(0,np.amax(l)+1):
        for j in range(i,np.amax(l)+1):
            Ek_sum[i]+=Ek[j]

    return [Ek_sum,ks]


def keSpectral_advective_flux_dp(u,v,pU,pD):
    zero=np.copy(u)*0.

    # get vorticity, divergence in spectral space 
    vrtsp,divsp = x.getvrtdivspec(u,v)
    vrt=x.spectogrd(vrtsp)
    div=x.spectogrd(divsp)

    # pressure Up is pi+1
    # pressure Down pi-1
    # for pressure weighted ke flux
    dp=pU-pD

    uk      = x.grdtospec(u)
    vk      = x.grdtospec(v)
    ux, uy  = x.getgrad(uk) # get gradients in grid space
    vx, vy  = x.getgrad(vk) # get gradients in grid space
    uak = x.grdtospec(dp*ux*u + dp*uy*v)  # advection, only
    vak = x.grdtospec(dp*vx*u + dp*vy*v)

    Usp = -1.*uak*uk.conj()
    Vsp = -1.*vak*vk.conj()
    Esp = Usp + Vsp 

    # build spectrum, initialize arrays
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    ks = np.arange(np.amax(l)+1)

    # select wavenumbers 
    for i in range(0,np.amax(l)+1):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])

    # running sum (Integral)
    for i in range(0,np.amax(l)+1):
        for j in range(i,np.amax(l)+1):
            Ek_sum[i]+=Ek[j]

    return [Ek_sum,ks]



# see Appendix, Equation B2
def CrossSpectral_flux_dp(u,v,pU,pD):
    # pressure Up is pi+1
    # pressure Down pi-1
    # for pressure weighted ke flux
    dp=pU-pD

    vrtsp,divsp = x.getvrtdivspec(u,v)
    u_vrt,v_vrt = x.getuv(vrtsp,divsp*0.)
    u_div,v_div = x.getuv(vrtsp*0.,divsp)

    uk_vrt = x.grdtospec(u_vrt) # intergate planetary vorticity in calculation
    vk_vrt = x.grdtospec(v_vrt)
    uk_div = x.grdtospec(u_div)
    vk_div = x.grdtospec(v_div)

    ux_vrt, uy_vrt = x.getgrad(uk_vrt) # get gradients in grid space
    vx_vrt, vy_vrt = x.getgrad(vk_vrt) # get gradients in grid space
    ux_div, uy_div = x.getgrad(uk_div) # get gradients in grid space
    vx_div, vy_div = x.getgrad(vk_div) # get gradients in grid space

    advecting_cross_terms_row1 = x.grdtospec(dp*ux_vrt*u_div + dp*uy_vrt*v_div) #cross advection of vortical and divergent terms
    advecting_cross_terms_row2 = x.grdtospec(dp*vx_vrt*u_div + dp*vy_vrt*v_div)
    advecting_cross_terms_row3 = x.grdtospec(dp*ux_div*u_vrt + dp*uy_div*v_vrt)
    advecting_cross_terms_row4 = x.grdtospec(dp*vx_div*u_vrt + dp*vy_div*v_vrt)
    advecting_cross_terms_row5 = x.grdtospec(dp*ux_vrt*u_vrt + dp*uy_vrt*v_vrt) #advecting cross terms
    advecting_cross_terms_row6 = x.grdtospec(dp*vx_vrt*u_vrt + dp*vy_vrt*v_vrt)
    advecting_cross_terms_row7 = x.grdtospec(dp*ux_div*u_div + dp*uy_div*v_div)
    advecting_cross_terms_row8 = x.grdtospec(dp*vx_div*u_div + dp*vy_div*v_div)

    asp = uk_vrt.conj()*advecting_cross_terms_row1 \
         +vk_vrt.conj()*advecting_cross_terms_row2 \
         +uk_div.conj()*advecting_cross_terms_row3 \
         +vk_div.conj()*advecting_cross_terms_row4 \
         +uk_div.conj()*advecting_cross_terms_row1 \
         +vk_div.conj()*advecting_cross_terms_row2 \
         +uk_vrt.conj()*advecting_cross_terms_row3 \
         +vk_vrt.conj()*advecting_cross_terms_row4 \
         +uk_div.conj()*advecting_cross_terms_row5 \
         +vk_div.conj()*advecting_cross_terms_row6 \
         +uk_vrt.conj()*advecting_cross_terms_row7 \
         +vk_vrt.conj()*advecting_cross_terms_row8
    Esp = -1.*asp

    # build spectrum, initialize arrays
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)

    # select wavenumbers 
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])

    # running sum (Integral)
    for i in range(0,np.amax(l)):
        for j in range(i,np.amax(l)):
            Ek_sum[i]+=Ek[j]

    return [Ek_sum,k]




# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
l = x._shtns.l
print("l:",l)


lons,lats = np.meshgrid(x.lons, x.lats)
latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)

folder='Output.HeldSuarez.HighResolRun'
path = '/home/scoty/miniGCM/Output.HeldSuarez.704ff582r15b/Fields/'


#write out for timesteps (from,to,in steps)
time=np.arange(0,381,1)
print('time.shape',time.shape)
print('time',time)
ke=np.zeros((3,time.shape[0]))

dPressure=750. # pressure difference between p1 and <ps>_area_mean [hPa]
p1=250.; p2=500.; p3=750. # [hPa]

icount=0
for it in time:
    print('it ',it)

    for Layer in np.arange(0,3):
        print("day ", it," Layer ",Layer)
        ps=netCDF4.Dataset(path+'Pressure_'+str(it*3600*24)+'.nc').variables['Pressure'][:,:]
        u=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer]
        v=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer]
        T=netCDF4.Dataset(path+'Temperature_'+str(it*3600*24)+'.nc').variables['Temperature'][:,:,Layer]
        Geo=netCDF4.Dataset(path+'Geopotential_'+str(it*3600*24)+'.nc').variables['Geopotential'][:,:,Layer]
        Wp=netCDF4.Dataset(path+'Wp_'+str(it*3600*24)+'.nc').variables['Wp'][:,:,Layer-1]/100.#hPa
        Ws=netCDF4.Dataset(path+'Wp_'+str(it*3600*24)+'.nc').variables['Wp'][:,:,2]/100.#hPa

        if Layer == 0:
            pD=ps*0.+p1
            pU=ps*0.+p2
            print('pU ',pU[1,1],' pD',pD[1,1],' Layer',Layer)
        elif Layer == 1:
            pD=ps*0.+p2
            pU=ps*0.+p3
            print('pU ',pU[1,1],' pD',pD[1,1],' Layer',Layer)
        elif Layer == 2:
            pD=ps*0.+p3
            pU=ps/100.
            print('pU ',pU[1,1],' pD',pD[1,1],' Layer',Layer)
        #

        iplot_spctr=1
        if (iplot_spctr==1):
           #
           # Energy & Enstrophy Flux Spectra
           #
           #
           #[E_flux,ks] = Enstrophy_flux(u,v) # enstrophy
           [KE_flux,ks] = keSpectral_flux(u,v,Geo,pU,pD) # energy
           if Layer==1 or Layer==2:
              uD=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer-1]
              vD=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer-1]
              print('Wp min ', np.amax(Wp), 'max', np.amin(Wp),' Layer',Layer)
              [KE_flux_error,ks] = keSpectral_flux_error_dp(u,v,Geo,pU,pD,uD,vD,Wp)
              np.save('Ek_flux_error_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KE_flux_error)
           if Layer==2:
              uD=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer]
              vD=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer]
              print('Wp min ', np.amax(Wp), 'max', np.amin(Wp),' Layer',Layer)
              [KE_flux_error,ks] = keSpectral_flux_error_dp(u*0.,v*0.,Geo,pU,pD,uD,vD,Ws)
              np.save('Ek_flux_error_'+str(Layer+1)+'_'+str(it).zfill(10)+'.npy',KE_flux_error)

           # energy flux divided in rotational, divergent, and cross-flux components
           [KE_advective_flux] = keSpectral_advective_flux_dp(u,v,pU,pD)
           [Cross_flux,ks] = CrossSpectral_flux_dp(u,v,pU,pD)
           [KErot_flux,ks] = keSpectral_advective_flux_dp(u_vrt,v_vrt,pU,pD)
           [KEdiv_flux,ks] = keSpectral_advective_flux_dp(u_div,v_div,pU,pD)

           # enertgy spectra, also divided in divergent and rotational components
           [EkTot,EkRot,EkDiv,ks] = energy(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True),l)
           np.save('EkTot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkTot)
           np.save('EkRot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkRot)
           np.save('EkDiv_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkDiv)
           #np.save('Enstrophy_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',E_flux)
           np.save('Ek_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KE_flux)
           np.save('Ek_advective_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KE_advective_flux)
           np.save('EkRot_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KErot_flux)
           np.save('EkDiv_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KEdiv_flux)
           np.save('EkCross_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',Cross_flux)
           np.save('ks.npy',ks)
        #
    icount+=1
    print('icount ',icount)
