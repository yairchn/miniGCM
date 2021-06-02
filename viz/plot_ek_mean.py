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

nlons  = hres_scaling*256  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)   # number of lats for gaussian grid.

# parameters for test
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5  # rotation rate
grav    = 9.80616   # gravity
Hbar    = 300.      # mean height (some typical range)

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
l = x._shtns.l
print("l:",l)


lons,lats = np.meshgrid(x.lons, x.lats)
latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)


runname='ftpgne60n256'
#runname='ftpgne.1n256'
#runname='f2800e.1n256'
path = '/home/suhas/miniGCM/Output.HeldSuarez.'+runname+'/Fields/'

it=200
Layer=0
Ek=np.load('EkTot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
print('Ek.shape ', Ek.shape)
EkTot=np.zeros((5,Ek.shape[0]))
EkRot=np.zeros((5,Ek.shape[0]))
EkDiv=np.zeros((5,Ek.shape[0]))
print('EkTot.shape ', EkTot.shape)

icount=1
for it in np.arange(220,801,20):
    print('it ',it)
    icount+=1

    for Layer in np.arange(0,5):
        print("day ", it," Layer ",Layer)
        #u=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer]
        #v=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer]
        #T=netCDF4.Dataset(path+'Temperature_'+str(it*3600*24)+'.nc').variables['Temperature'][:,:,Layer]

        #print('u', u.shape)
        #print('v', v.shape)

        iplot_spctr=1
        if (iplot_spctr==1):
           #
           # Energy Spectra
           #
           #
           plt.figure(33)
           plt.clf()
           #[KE,ks] = keSpectra(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True))
           #plt.loglog(ks,savgol_filter(KE,5,1))
           #[EkTot,EkRot,EkDiv,ks] = energy(u-u.mean(axis=1, keepdims=True),v-v.mean(axis=1, keepdims=True),l)
           EkTot[Layer,:]+=np.load('EkTot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
           EkRot[Layer,:]+=np.load('EkRot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
           EkDiv[Layer,:]+=np.load('EkDiv_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
           ks=np.load('ks.npy')

EkTot/=float(icount)
EkRot/=float(icount)
EkDiv/=float(icount)


for Layer in np.arange(0,5):
    plt.loglog(ks,1.e2*ks**(-5./3.),'--k',linewidth=2)
    plt.loglog(ks,1.e5*ks**(-3.),'-k',linewidth=2)
    plt.title('KE Spectra')
    plt.grid()
    plt.ylim(1.e-6,1.e4)
    plt.savefig('Ek_'+str(Layer)+'_mean.png')
    #


    fig,ax = plt.subplots(1,5, sharey='all',figsize=(25, 5),squeeze=False)

    i = 0
    li=[4,3,2,1,.5]
    al=[0.4,0.6,0.8,0.9,1.]
    la=['125 hPa','250 hPa','500 hPa','750 hPa','850 hPa']
    for j in range(5):
        ax[i,j].loglog(ks,EkTot[j,:],'-k',alpha=0.4,linewidth=4, label='EK')
        ax[i,j].loglog(ks,EkRot[j,:],'-r', label='EK rot')
        ax[i,j].loglog(ks,EkDiv[j,:],'-b', label='EK div')
        ax[i,j].loglog(ks,1.e2*ks**(-5./3.),'--k',linewidth=2,label='-5/3')
        ax[i,j].loglog(ks,1.e5*ks**(-3.),'-k',linewidth=2,label='-3')
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5],linewidth=1,zorder=10)
        #ax[i,j].set_yticks(np.linspace(-90,90,7))
        ax[i,j].set_ylim(1.e-6,1.e4)
        #ax[i,j].set_xlim(-1,1)
        #ax[i,j].axvline(x=0,color='k',linestyle='dashed',zorder=25,alpha=0.75)
        #ax[i,j].ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax[i,j].set_xlabel('Wavenumber k',size='12', fontname = 'Dejavu Sans')
        ax[i,j].set_title('Layer '+str(j+1), size='12', fontname = 'Dejavu Sans')

    ax[0,0].legend(loc='upper right')
    ax[0,0].set_ylabel('Latitude / $\circ$',size='12', fontname = 'Dejavu Sans')

    plt.tight_layout()
    #plt.suptitle("Eddy Kinetic Energy / m$^2$ s$^{-2}$",size='14', fontname = 'Dejavu Sans')
    plt.savefig('eke_5layers_'+runname+'.pdf', bbox_inches='tight',dpi=150)


