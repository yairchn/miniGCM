#Code to solve moist shallow water equations on a sphere
#SWE are of the vorticity-divergence form.
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import scipy as sc
import xarray
import netCDF4
import seaborn
from scipy.signal import savgol_filter
from scipy.interpolate import interp2d


import sys

seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})
#plt.ion()
###########################################

# hres_scaling=4
hres_scaling=1

nlons  = hres_scaling*512  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)   # number of lats for gaussian grid.

# parameters for test
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5  # rotation rate
grav    = 9.80616   # gravity
Hbar    = 300.      # mean height (some typical range)

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lats) # coriolis

# Relaxation time scales
#tau_vrt = 10*24*3600.
#tau_div = 10*24*3600.
#tau_ht = 10*24*3600.

#tau_q   = 10*24*3600.   # evaporation time scale
#tau_c   = 0.1*24*3600.  # condensation time scale

#chi = 5500.  #### coversion factor similar to latent heat
# make sure that (h - chi*Qsat) is always positive

latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)
l = x._shtns.l
print("l:",l)

def energy(u,v,l,rsphere= rsphere):
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

def keSpectra(u,v):
    uk = x.grdtospec(u)
    vk = x.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
    return [Ek,k]





##################################################################
# setting up the integration
tmax = 10.01*24*3600  #(time to integrate, here 10 days)
tmax = 15.01*24*3600  #(time to integrate, here 15 days)
tmax = 30.01*24*3600  #(time to integrate, here 30 days)
################################################################
latdim = nlats
londim = nlons
dims = [latdim,londim]
dim_names = ['latitude','longitude']

path     = './'  # path to save the file
path='/DATA/suhas/sw/'
filename = 'testMoistStrongVrtForcedWithEvapHeatWeakerForcing_Hbar100m_L1'


fields   = ['uwind','vwind','vorticity','divergence','height']
# variables to store (this only saves the data for a field which is a fn of lat, lon and time).
cords    = [x.lats,x.lons]

#const_name = ['radius','rotation','gravity', 'chi'] #for constants
#const_data = [rsphere,omega,grav, chi]



dataset = netCDF4.Dataset(path+filename+'.nc')
nt =dataset.variables['vorticity'][:,0,0].shape[0]
print('nt =', nt)



for it in np.arange(100,1000,100):
    print("it=", it)
    div=dataset.variables['divergence'][it,:,:]
    vort=dataset.variables['vorticity'][it,:,:]
    u=dataset.variables['uwind'][it,:,:]
    v=dataset.variables['vwind'][it,:,:]
    ht=dataset.variables['height'][it,:,:]
    #qp=dataset.variables['qprime'][it,:,:]

    print("ht.shape", ht.shape)
    print("ht mean", np.mean(ht))
    print("lonDeg.shape", lonDeg.shape)
    print("latDeg.shape", latDeg.shape)

    latDeg0=latDeg[:,0]
    lonDeg0=lonDeg[0,:]

    speed=np.sqrt(u*u + v*v)

    lw = np.sqrt(speed / speed.max())


    #
    # histograms ---
    #
    plt.figure(87,figsize=(4,4))
    plt.clf()
    abins=np.arange(np.amin(u),np.amax(u),.5)
    histo,ebins=np.histogram(u,bins=abins)
    plt.semilogy(ebins[0:ebins.shape[0]-1],histo)
    plt.savefig('u_histo'+filename+'_'+str(it).zfill(10)+'.png')

    plt.figure(88,figsize=(4,4))
    plt.clf()
    abins=np.arange(np.amin(v),np.amax(v),.5)
    histo,ebins=np.histogram(v,bins=abins)
    plt.semilogy(ebins[0:ebins.shape[0]-1],histo)
    plt.savefig('v_histo'+filename+'_'+str(it).zfill(10)+'.png')

    #
    # z o n a l -- m e a n s
    #
    zonalmean=u.mean(axis=1)
    zonalstd=u.std(axis=1)
    #print("zonal mean of u, shape: ", zonalmean, zonalmean.shape)
    maxAnomaly = np.amax(abs(zonalmean))
    plt.figure(7,figsize=(4,8))
    plt.clf()
    plt.title("Zonal means at day "+str(it))
    plt.plot(zonalmean,latDeg,'-k',linewidth='2')
    plt.plot(zonalmean-zonalstd,latDeg,'-k')
    plt.plot(zonalmean+zonalstd,latDeg,'-k')
    plt.savefig('u_zonalmean_'+filename+'_'+str(it).zfill(10)+'.png')
    np.save('latDeg.npy',latDeg0)
    print("zonalmean.shape", zonalmean.shape)
    np.save('u_zonalmean_'+filename+'_'+str(it).zfill(10)+'.npy',np.array(zonalmean))
    np.save('u_zonalstd_'+filename+'_'+str(it).zfill(10)+'.npy',np.array(zonalstd))

    zonalmean=v.mean(axis=1)
    zonalstd=v.std(axis=1)
    #print("zonal mean of v: ", zonalmean, zonalmean.shape)
    maxAnomaly = np.amax(abs(zonalmean))
    plt.figure(7,figsize=(4,8))
    plt.clf()
    plt.title("Zonal means at day "+str(it))
    plt.plot(zonalmean,latDeg,'-k',linewidth='2')
    plt.plot(zonalmean-zonalstd,latDeg,'-k')
    plt.plot(zonalmean+zonalstd,latDeg,'-k')
    plt.savefig('v_zonalmean_'+filename+'_'+str(it).zfill(10)+'.png')
    #
    zonalmean=0.5*(u*u + v*v).mean(axis=1)
    zonalstd=0.5*(u*u + v*v).std(axis=1)
    #print("zonal mean of ekin: ", zonalmean, zonalmean.shape)
    maxAnomaly = np.amax(abs(zonalmean))
    plt.figure(7,figsize=(4,8))
    plt.clf()
    plt.title("Zonal means at day "+str(it))
    plt.plot(zonalmean,latDeg,'-k',linewidth='2')
    plt.plot(zonalmean-zonalstd,latDeg,'-k')
    plt.plot(zonalmean+zonalstd,latDeg,'-k')
    plt.savefig('ekin_zonalmean_'+filename+'_'+str(it).zfill(10)+'.png')
    #

    zonalmean=(ht).mean(axis=1)
    zonalstd=(ht).std(axis=1)
    #print("zonal mean of h: ", zonalmean, zonalmean.shape)
    print("mean of h: ", np.mean(ht))
    maxAnomaly = np.amax(abs(zonalmean))
    plt.figure(7,figsize=(4,8))
    plt.clf()
    plt.title("Zonal means at day "+str(it))
    plt.plot(zonalmean,latDeg,'-k',linewidth='2')
    plt.plot(zonalmean-zonalstd,latDeg,'-k')
    plt.plot(zonalmean+zonalstd,latDeg,'-k')
    plt.savefig('h_zonalmean_'+filename+'_'+str(it).zfill(10)+'.png')

    #
    # c o n t o u r s
    #
    plt.figure(6,figsize=(8,4))
    plt.clf()
    maxAnomaly = np.amax(abs(v))
    print("maxV: ",maxAnomaly)
    levels = np.linspace(-8, 8, 100)
    plt.ylim([-45, 45])
    plt.title("Meridonial velocity [m/s] at day "+str(it))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.contourf(lonDeg,latDeg,v,levels,extend='both',cmap='coolwarm')
    plt.colorbar(orientation='horizontal',extend="both")
    plt.savefig('v_'+filename+'_'+str(it).zfill(10)+'.png')

    plt.figure(6,figsize=(8,4))
    plt.clf()
    maxAnomaly = np.amax(abs(u))
    print("maxU: ",maxAnomaly)
    levels = np.linspace(-8, 8, 100)
    plt.ylim([-45, 45])
    plt.title("Zonal velocity [m/s] at day "+str(it))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.contourf(lonDeg,latDeg,u,levels,extend='both',cmap='coolwarm')
    plt.colorbar(orientation='horizontal',extend="both")
    #plt.ylim(-25,25)
    #plt.tight_layout()
    #plt.savefig('u_colorbar_'+filename+'_'+str(it).zfill(2))
    plt.savefig('u_'+filename+'_'+str(it).zfill(10)+'.png')


    #
    # Energy Spectra
    #
    plt.figure(3)
    plt.clf()
    [EK,ks] = keSpectra(u,v)
    plt.loglog(ks,savgol_filter(EK,5,1))
    plt.title('KE Spectra')
    plt.grid()
    plt.savefig('Ek_full_'+filename+'_'+str(it).zfill(10)+'.png')
    np.save('Ek_full_'+filename+'_'+str(it).zfill(10)+'.npy',savgol_filter(EK,5,1))
    np.save('Ek_full_'+filename+'_'+str(it).zfill(10)+'_ks.npy',ks)
    #
    #
    #
    plt.figure(33)
    plt.clf()
    [EK,ks] = keSpectra(u-u.mean(axis=1, keepdims=True),v)
    plt.loglog(ks,savgol_filter(EK,5,1))
    plt.loglog(ks,1.e2*ks**(-5./3.))
    #[TotEk,RotEk,DivEk,k] = energy(u,v,l)
    #plt.loglog(k,TotEk)
    #plt.loglog(k,RotEk)
    #plt.loglog(k,DivEk)
    plt.title('KE Spectra')
    #plt.ylim(1.e-2,1.e+2)
    plt.grid()
    plt.savefig('Ek_'+filename+'_'+str(it).zfill(10)+'.png')
    np.save('Ek_'+filename+'_'+str(it).zfill(10)+'.npy',savgol_filter(EK,5,1))
    np.save('Ek_'+filename+'_'+str(it).zfill(10)+'_ks.npy',ks)
    #


    plt.figure(2,figsize=(8,4))
    plt.clf()
    # maxAnomaly = np.amax(abs(ht))
    # minAnomaly = np.amin(abs(ht))
    maxAnomaly = 110
    minAnomaly = 90
    levels = np.linspace(minAnomaly, maxAnomaly, 100)
    plt.contourf(lonDeg,latDeg,ht,levels,extend='both',cmap='coolwarm')
    plt.ylim([-45, 45])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(orientation='horizontal',aspect=50,extend="both")#,ticks=[223.,224.,225.,226.,227.,228.,229.,230.])
    # regularly spaced grid spanning the domain of x and y
    xi = np.linspace(lonDeg.min(), lonDeg.max(), lonDeg0.size)
    yi = np.linspace(latDeg.min(), latDeg.max(), latDeg0.size)
    xx  = lonDeg0
    yy  = latDeg0
    # bicubic interpolation
    uCi  = interp2d(xx, yy,  u)(xi, yi)
    vCi  = interp2d(xx, yy,  v)(xi, yi)
    lwCi = interp2d(xx, yy, lw)(xi, yi)
    #
    plt.streamplot(xi, yi, uCi, vCi, density=0.8, color='k', linewidth=lwCi)
    plt.title('h - Shallow water height [m] at day ' + str(it))
    plt.grid()
    plt.tight_layout()
    plt.savefig('ht_'+filename+'_'+str(it).zfill(10)+'.png')

    #plt.figure(22,figsize=(8,3.5))
    #plt.clf()
    #maxAnomaly = np.amax(qp)
    #minAnomaly = np.amin(qp)
    #minAnomaly = -1.e-3
    #maxAnomaly = +1.e-3
    #print("qp min max ",minAnomaly,maxAnomaly)
    #levels = np.linspace(minAnomaly, maxAnomaly, 100)
    #plt.contourf(lonDeg,latDeg,qp,levels,extend='both',cmap='Blues')
    #plt.colorbar(orientation='horizontal',aspect=50,extend="both",ticks=np.arange(-1.e-3,1.e-3,.2e-3))
    ##plt.streamplot(lonDeg[10:-10,10:-10], latDeg[10:-10,10:-10], u[10:-10,10:-10], v[10:-10,10:-10], density=0.8, color='k', linewidth=lw[10:-10,10:-10])
    #plt.title('qp')
    ##plt.ylim((-40,40))
    #plt.ylim((-60,60))
    #plt.grid()
    #plt.tight_layout()
    #plt.savefig('qp_'+filename+'_'+str(it).zfill(10)+'.png')
