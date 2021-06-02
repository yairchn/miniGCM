#Code to solve moist shallow water equations on a sphere
#SWE are of the vorticity-divergence form.
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import netCDF4
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

hres_scaling=4
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


def keSpectra(u,v):
    uk = x.grdtospec(u)
    vk = x.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
    return [Ek,k]



# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
l = x._shtns.l
print("l:",l)


lons,lats = np.meshgrid(x.lons, x.lats)
latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)

folder='Output.HeldSuarez.ReferenceRun'
path = '/home/scoty/miniGCM/'+folder+'/Fields/'


for it in np.arange(504,560,7):
    print('it ',it)

    for Layer in np.arange(0,3):
        print("day ", it," Layer ",Layer)
        u=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer]
        v=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer]
        T=netCDF4.Dataset(path+'Temperature_'+str(it*3600*24)+'.nc').variables['Temperature'][:,:,Layer]

        print('u', u.shape)
        print('v', v.shape)

        iplot_mean=1
        if (iplot_mean==1):

           zonalmean=u.mean(axis=1)
           zonalstd=u.std(axis=1)
           #print("zonal mean of u: ", zonalmean, zonalmean.shape)
           maxAnomaly = np.amax(abs(zonalmean))
           plt.figure(3,figsize=(4,8))
           plt.clf()
           plt.title("Zonal means at day "+str(it))
           plt.plot(zonalmean,latDeg,'-k',linewidth='2')
           plt.plot(zonalmean-zonalstd,latDeg,'-k')
           plt.plot(zonalmean+zonalstd,latDeg,'-k')
           plt.savefig('u_zonalmean_'+str(Layer)+'_'+str(it).zfill(10)+'.png')

           zonalmean=v.mean(axis=1)
           zonalstd=v.std(axis=1)
           #print("zonal mean of v: ", zonalmean, zonalmean.shape)
           maxAnomaly = np.amax(abs(zonalmean))
           plt.figure(3,figsize=(4,8))
           plt.clf()
           plt.title("Zonal means at day "+str(it))
           plt.plot(zonalmean,latDeg,'-k',linewidth='2')
           plt.plot(zonalmean-zonalstd,latDeg,'-k')
           plt.plot(zonalmean+zonalstd,latDeg,'-k')
           plt.savefig('v_zonalmean_'+str(Layer)+'_'+str(it).zfill(10)+'.png')
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
           plt.savefig('ekin_zonalmean_'+str(Layer)+'_'+str(it).zfill(10)+'.png')
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


        iplot_spctr=0
        if (iplot_spctr==1):
           #
           # Energy Spectra
           #
           #
           #plt.figure(33)
           #plt.clf()
           #[KE,ks] = keSpectra(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True))
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
           np.save('EkTot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkTot)
           np.save('EkRot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkRot)
           np.save('EkDiv_'+str(Layer)+'_'+str(it).zfill(10)+'.npy',EkDiv)
           np.save('ks.npy',ks)
        #

