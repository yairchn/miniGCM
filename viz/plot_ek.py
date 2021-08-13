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

#some different way of calculating spectra that is currently inactive
def keSpectra(u,v):
    uk = x.grdtospec(u)
    vk = x.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
    return [Ek,k]


def Enstrophy_flux(u,v):
    vrtsp,divsp = x.getvrtdivspec(u,v)

    vrt_x, vrt_y = x.getgrad(vrtsp) # get gradients in grid space

    vrtsp_advection = x.grdtospec(vrt_x*u + vrt_y*v)

    Enstrophy_flux = -1.*vrtsp*vrtsp_advection.conj()

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

def keSpectral_flux(u,v):
    uk = x.grdtospec(u)
    vk = x.grdtospec(v)
    ux, uy = x.getgrad(uk) # get gradients in grid space
    vx, vy = x.getgrad(vk) # get gradients in grid space
    uak = x.grdtospec(ux*u + uy*v)
    vak = x.grdtospec(vx*u + vy*v)


    Usp = -1.*uak*uk.conj()
    Vsp = -1.*vak*vk.conj()
    Esp = Usp + Vsp

    # build spectrum
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)

    # running sum (Integral)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
    for i in range(0,np.amax(l)):
        for j in range(i,np.amax(l)):
            Ek_sum[i]+=Ek[j]

    return [Ek_sum,k]

def CrossSpectral_flux(u,v):
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

    advecting_cross_terms_row1 = x.grdtospec(ux_vrt*u_div + uy_vrt*v_div)
    advecting_cross_terms_row2 = x.grdtospec(vx_vrt*u_div + vy_vrt*v_div)
    advecting_cross_terms_row3 = x.grdtospec(ux_div*u_vrt + uy_div*v_vrt)
    advecting_cross_terms_row4 = x.grdtospec(vx_div*u_vrt + vy_div*v_vrt)

    asp = uk_vrt.conj()*advecting_cross_terms_row1 \
         +vk_vrt.conj()*advecting_cross_terms_row2 \
         +uk_div.conj()*advecting_cross_terms_row3 \
         +vk_div.conj()*advecting_cross_terms_row4

    Esp = -1.*asp

    # build spectrum
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)

    # running sum (Integral)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
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
path = '/home/josefs/miniGCM/'+folder+'/Fields/'
path = '/home/scoty/miniGCM/Output.HeldSuarez.ReferenceRun/Fields_restart_factor4/'
path = '/home/josefs/miniGCM/Output.HeldSuarez.JustAtestRun/Fields_restart_factor8/'
path = '/home/josefs/miniGCM/Output.HeldSuarez.JustAtestRun/Fields_restart_factor16/'
path = '/home/scoty/miniGCM/Output.HeldSuarez..44-test3lay/Fields/'


eke=np.zeros((3,801))
time=np.arange(0,801)


#for it in np.arange(500,801,5):
for it in np.arange(0,801,1):
    print('it ',it)

    for Layer in np.arange(0,3):
        print("day ", it," Layer ",Layer)
        u=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer]
        v=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer]
        T=netCDF4.Dataset(path+'Temperature_'+str(it*3600*24)+'.nc').variables['Temperature'][:,:,Layer]

        vrtsp,divsp = x.getvrtdivspec(u,v)
        u_vrt,v_vrt = x.getuv(vrtsp,divsp*0.)
        u_div,v_div = x.getuv(vrtsp*0.,divsp)

        print('u', u.shape)
        print('v', v.shape)
        print('T', T.shape)

        iplot_timsr=1
        if (iplot_timsr==1):
           eke[Layer,it]=np.mean(0.5*(u-u.mean(axis=1,keepdims=True))**2 + 0.5*(v-v.mean(axis=1,keepdims=True))**2)

        iplot_mean=0
        if (iplot_mean==1):

           zonalmean=T.mean(axis=1)
           #zonalstd=T.std(axis=1)
           #print("zonal mean of T: ", zonalmean, zonalmean.shape)
           #maxAnomaly = np.amax(abs(zonalmean))
           #plt.figure(3,figsize=(4,8))
           #plt.clf()
           #plt.title("Zonal means at day "+str(it))
           #plt.plot(zonalmean,latDeg,'-k',linewidth='2')
           #plt.plot(zonalmean-zonalstd,latDeg,'-k')
           #plt.plot(zonalmean+zonalstd,latDeg,'-k')
           #plt.savefig('T_zonalmean_'+str(Layer)+'_'+str(it).zfill(10)+'.png')
           np.save('T_zonalmean_'+str(Layer)+'_'+str(it).zfill(10),np.array(zonalmean))


           zonalmean=u.mean(axis=1)
           #zonalstd=u.std(axis=1)
           #print("zonal mean of u: ", zonalmean, zonalmean.shape)
           #maxAnomaly = np.amax(abs(zonalmean))
           #plt.figure(3,figsize=(4,8))
           #plt.clf()
           #plt.title("Zonal means at day "+str(it))
           #plt.plot(zonalmean,latDeg,'-k',linewidth='2')
           #plt.plot(zonalmean-zonalstd,latDeg,'-k')
           #plt.plot(zonalmean+zonalstd,latDeg,'-k')
           #plt.savefig('u_zonalmean_'+str(Layer)+'_'+str(it).zfill(10)+'.png')
           np.save('u_zonalmean_'+str(Layer)+'_'+str(it).zfill(10),np.array(zonalmean))

           zonalmean=v.mean(axis=1)
           #zonalstd=v.std(axis=1)
           #print("zonal mean of v: ", zonalmean, zonalmean.shape)
           #maxAnomaly = np.amax(abs(zonalmean))
           #plt.figure(3,figsize=(4,8))
           #plt.clf()
           #plt.title("Zonal means at day "+str(it))
           #plt.plot(zonalmean,latDeg,'-k',linewidth='2')
           #plt.plot(zonalmean-zonalstd,latDeg,'-k')
           #plt.plot(zonalmean+zonalstd,latDeg,'-k')
           #plt.savefig('v_zonalmean_'+str(Layer)+'_'+str(it).zfill(10)+'.png')
           np.save('v_zonalmean_'+str(Layer)+'_'+str(it).zfill(10),np.array(zonalmean))
           np.save('latDeg',latDeg)
           #
           #zonalmean=0.5*(u*u + v*v).mean(axis=1)
           #zonalstd=0.5*(u*u + v*v).std(axis=1)
           #print("zonal mean of ekin: ", zonalmean, zonalmean.shape)
           #maxAnomaly = np.amax(abs(zonalmean))
           #plt.figure(7,figsize=(4,8))
           #plt.clf()
           #plt.title("Zonal means at day "+str(it))
           #plt.plot(zonalmean,latDeg,'-k',linewidth='2')
           #plt.plot(zonalmean-zonalstd,latDeg,'-k')
           #plt.plot(zonalmean+zonalstd,latDeg,'-k')
           #plt.savefig('ekin_zonalmean_'+str(Layer)+'_'+str(it).zfill(10)+'.png')
           #

        iplot_ctr=0
        if (iplot_ctr==1):
           #
           # c o n t o u r s
           #
           plt.figure(4,figsize=(6,2))
           plt.clf()
           maxAnomaly = np.amax(abs(v))
           #print("maxV: ",maxAnomaly)
           levels = np.linspace(-28, 28, 40)
           plt.ylim([-45, 45])
           plt.title("Meridonial velocity [m/s] at day "+str(it))
           plt.xlabel("Longitude")
           plt.ylabel("Latitude")
           plt.contourf(lonDeg,latDeg,v,levels,extend='both',cmap='coolwarm')
           #plt.colorbar(orientation='horizontal',extend="both")
           plt.tight_layout()
           plt.savefig('v_'+str(Layer)+'_'+str(it).zfill(10)+'.png')

           plt.figure(6,figsize=(6,2))
           plt.clf()
           maxAnomaly = np.amax(abs(u))
           #print("maxU: ",maxAnomaly)
           levels = np.linspace(-8, 38, 40)
           plt.ylim([-45, 45])
           plt.title("Zonal velocity [m/s] at day "+str(it))
           plt.xlabel("Longitude")
           plt.ylabel("Latitude")
           plt.contourf(lonDeg,latDeg,u,levels,extend='both',cmap='coolwarm')
           #plt.colorbar(orientation='horizontal',extend="both")
           plt.tight_layout()
           plt.savefig('u_'+str(Layer)+'_'+str(it).zfill(10)+'.png')

           plt.figure(7,figsize=(6,3))
           plt.clf()
           maxAnomaly = np.amax(abs(T))
           minAnomaly = np.amin(abs(T))
           #print("maxT: ",maxAnomaly)
           levels = np.linspace(minAnomaly, maxAnomaly, 40)
           plt.ylim([-65, 65])
           plt.title("Temperature [K] at day "+str(it))
           plt.xlabel("Longitude")
           plt.ylabel("Latitude")
           plt.contourf(lonDeg,latDeg,T,levels,extend='both',cmap='coolwarm')
           plt.colorbar(orientation='horizontal',extend="both")
           plt.tight_layout()
           plt.savefig('T_'+str(Layer)+'_'+str(it).zfill(10)+'.png')


        iplot_spctr=1
        if (iplot_spctr==1):
           #
           # Energy Spectra
           #
           #
           #plt.figure(33)
           #plt.clf()
           [E_flux,ks] = Enstrophy_flux(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True))
           [KE,ks] = keSpectra(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True))
           [KE_flux,ks] = keSpectral_flux(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True))
           [Cross_flux,ks] = CrossSpectral_flux(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True))
           [KErot_flux,ks] = keSpectral_flux(u_vrt-u_vrt.mean(axis=1, keepdims=True),v_vrt-v_vrt.mean(axis=1, keepdims=True))
           [KEdiv_flux,ks] = keSpectral_flux(u_div-u_div.mean(axis=1, keepdims=True),v_div-v_div.mean(axis=1, keepdims=True))
           #plt.loglog(ks,savgol_filter(KE,5,1))
           [EkTot,EkRot,EkDiv,ks] = energy(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True),l)
           #plt.loglog(ks,EkTot,'-k',alpha=0.4,linewidth=4)
           #plt.loglog(ks,EkRot,'-r')
           #plt.loglog(ks,EkDiv,'-b')
           #plt.loglog(ks,1.e2*ks**(-5./3.),'--k',linewidth=2)
           #plt.loglog(ks,1.e5*ks**(-3.),'-k',linewidth=2)
           #plt.title('KE Spectra')
           #plt.grid()
           #plt.ylim(1.e-6,1.e4)
           #plt.savefig('Ek_'+str(Layer)+'_'+str(it).zfill(10)+'.png')
           #np.save('Ek_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KE)
           np.save('EkTot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkTot)
           np.save('EkRot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkRot)
           np.save('EkDiv_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkDiv)
           #np.save('Ek_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KE_flux)
           #np.save('Enstrophy_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',E_flux)
           #np.save('EkRot_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KErot_flux)
           #np.save('EkDiv_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',KEdiv_flux)
           #np.save('EkCross_flux_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',Cross_flux)
           np.save('ks.npy',ks)
        #

iplot_timsr=1
if (iplot_timsr==1):
   zonalstd=T.std(axis=1)
   plt.figure(3,figsize=(5,2.5))
   plt.clf()
   plt.title("Global Mean Eddy Kinetic Energy")
   #plt.plot(time,eke[0,:],'-k',alpha=0.9,linewidth='1',label='top')
   #plt.plot(time,eke[1,:],'-k',alpha=0.7,linewidth='1',label='middle')
   #plt.plot(time,eke[2,:],'-k',alpha=0.5,linewidth='1',label='bottom')
   plt.plot(time,(eke[0,:]+eke[1,:]+eke[2,:])/3.,'-k',alpha=1.,linewidth=1,label='bottom')
   plt.tight_layout()
   plt.ylabel('eke / m$^2$ s$^{-2}$')
   plt.xlabel('time / days')
   plt.tight_layout()
   plt.savefig('eke_timeseries.pdf')


