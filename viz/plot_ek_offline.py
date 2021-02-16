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

hres_scaling=1

nlons  = hres_scaling*512  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)   # number of lats for gaussian grid.

# parameters for test
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5  # rotation rate
grav    = 9.80616   # gravity
Hbar    = 300.      # mean height (some typical range)

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



# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
l = x._shtns.l
print("l:",l)


lons,lats = np.meshgrid(x.lons, x.lats)
latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)


path='../Output.HeldSuarez.410k2/Fields/'


ks=np.load("ks.npy")
EkTot=np.copy(ks)*0.
EkRot=np.copy(ks)*0.
EkDiv=np.copy(ks)*0.

count=0
for it in np.arange(200,1000,40):

    Layer=1
    count+=1
    print("day ", it," Layer ",Layer, " Count",count)

    iplot_spctr=1
    if (iplot_spctr==1):
       #
       # Energy Spectra
       #
       #
       EkTot+=np.load('EkTot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
       EkRot+=np.load('EkRot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
       EkDiv+=np.load('EkDiv_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
    #
EkTot/=float(count)
EkRot/=float(count)
EkDiv/=float(count)

plt.figure(33,figsize=(5.,3.5))
plt.clf()
#[KE,ks] = keSpectra(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True))
#plt.loglog(ks,savgol_filter(KE,5,1))
plt.loglog(ks,EkTot,'-k',alpha=0.4,linewidth=4,label='Ek')
plt.loglog(ks,EkRot,'-r',label='EK rot')
plt.loglog(ks,EkDiv,'-b',label='EK div')
plt.loglog(ks[30:100],.6e3*ks[30:100]**(-5./3.),'--k',linewidth=2,label='-5/3')
plt.loglog(ks[1:40],1.e5*ks[1:40]**(-3.),'-k',linewidth=1,label='-3')
plt.title('KE Spectra')
plt.grid()
plt.ylim(1.e-4,1.e2)
plt.xlim(1,1.4e2)
plt.legend()
plt.ylabel('Eddy Kinetic Energy [m$^2$/s$^2$]')
plt.xlabel('Wavenumber')
plt.tight_layout()
plt.savefig('Ek_'+str(Layer)+'.png',dpi=150)



