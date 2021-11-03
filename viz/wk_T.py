import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import scipy as sc
import xarray
import netCDF4
import matplotlib.ticker as mticker
from numpy.fft import fft, ifft, fft2,ifft2,fftshift,fftfreq
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import seaborn
import dispersionCurves as dc
from scipy import signal

#exit()

hres=1
hres=4
nlons = 128*hres  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats = int(nlons/2)   # for gaussian grid.


# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5 # rotation rate
grav = 9.80616 # gravity

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
longi,lati = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lati)
factor = np.cos(x.lats)
Hbar = 400.
Hbar = 250.
Hbar = 200.
Hbar = 100.
#Hbar = 300.
#Hbar = 30.
Hbar = 60.

disp = dc.curves(lat=np.deg2rad(7.5), grav=grav,rsphere=rsphere,omega=omega,mean=0.)

def plotDispCurves(Hbar):
    freqKW,kKW   = disp.KW(Hbar)
    freqRW1,kRW1 = disp.ERW(Hbar,1.)
    freqRW2,kRW2 = disp.ERW(Hbar,2.)
    freqRW3,kRW3 = disp.ERW(Hbar,3.)
    freqMRG,kMRG = disp.MRG(Hbar)
 
    freqIGW1,kIGW1 = disp.IGW(Hbar,1.)
    freqIGW2,kIGW2 = disp.IGW(Hbar,2.)
    freqIGW3,kIGW3 = disp.IGW(Hbar,3.)
    freqIGW4,kIGW4 = disp.IGW(Hbar,4.)
    freqIGW5,kIGW5 = disp.IGW(Hbar,5.)
 
    plt.plot(kKW ,freqKW,':k',linewidth='1.',label='Kelvin')
    plt.plot(kRW1,freqRW1,'--k',linewidth='.25',label='Rossby')
    plt.plot(kRW2,freqRW2,'--k',linewidth='.25')
    plt.plot(kRW3,freqRW3,'--k',linewidth='.25')
    plt.plot(kMRG,freqMRG,'-k',linewidth='.5',label='Yanai')
    plt.plot(kIGW1,freqIGW1,'--k',linewidth='.5',alpha=0.5,label='IGW')
    plt.plot(kIGW2,freqIGW2,'--k',linewidth='.5',alpha=0.5)
    plt.plot(kIGW3,freqIGW3,'--k',linewidth='.5',alpha=0.5)
    plt.plot(kIGW4,freqIGW4,'--k',linewidth='.5',alpha=0.5)
    plt.plot(kIGW5,freqIGW5,'--k',linewidth='.5',alpha=0.5)

def calcFlux(name,tmin,tmax,lmin,lmax,dt,dx=1.,Layer=0):
    data = netCDF4.Dataset(name)
    data=data.variables['Temperature'][:,:,:,:]
    print("data.shape ", data.shape)
    #u    = data['uwind']
    #v    = data['vwind']
    #vort = data['vorticity']
    #field = data['Temperature'][500:800,28:36,:,2]-np.mean(data['Temperature'][500:800,28:36,:,2],axis=(0,2))
    field = np.array(data[0:300,28:36,:,Layer])

    #qp   = data['qprime']
    #vrt  = data['vorticity']
    #print("time[0] ", time[0])
    #ntim = len(time)
    #nlon = len(lon)
    #print("ntim, nlon", ntim, nlon)
    #print("tmin, tmax", tmin, tmax)

    #sel={}
    #sel['time'] = slice(tmin,tmax)
    #sel['time'] = slice(500,800,1)
    #sel['latitude'] = slice(np.deg2rad(lmax),np.deg2rad(lmin))
    ###vrtsel = vrt.loc[sel]
    #htsel = ht.loc[sel]
    out = field
    print("out.shape ", out.shape)
    #print("out ", out)
    #out   = vrtsel
    #taper with Hanning window
    taper= signal.windows.hann(out.shape[0])
    out  = out*taper[:,np.newaxis,np.newaxis]
    power= calcPower(out)

    ##########################################
    #lat  = out.latitude
    #lon  = out.longitude
    #time = out.time
    #ntim = len(time)
    #nlon = len(lon)
    #print("ntim, nlon", ntim, nlon)
    ntim=out.shape[0]
    dt=1.
    dx   = dx/nlons
    kxx  = fftfreq(nlons,dx)
    ktt  = fftfreq(ntim,dt)
    [kx,kt] = np.meshgrid(kxx,ktt)
    kx   = fftshift(kx)
    kt   = fftshift(kt)
    # power[abs(kx)<1] = 0.
    #power = power/np.amax(abs(power))
    #power = np.log10(power)
    return [power,kx,kt]

def calcPower(data):
    p     = fft2(data,axes=(0,2))
    power = (p*p.conj()).real
    power = np.mean(power,axis=1)
#    power = power/np.amax(abs(power))
    power = fftshift(power)
    return np.flip(power,axis=0)
########################################################

tmin =  500
tmax =  800
dt   = 1.

path     = '../Output.HeldSuarez..44-test3lay/Fields/'
path     = '../Output.HeldSuarez.oistp_plane6/Fields/'
filename = 'T_merge.nc'

#lmin= 45.; lmax=65.
lmin=-15.; lmax=15.
sp1     = calcFlux(path+filename,tmin,tmax,lmin,lmax,dt,Layer=0)
#sp2     = calcFlux(path+filename,tmin,tmax,lmin,lmax,dt,Layer=1)
#sp3     = calcFlux(path+filename,tmin,tmax,lmin,lmax,dt,Layer=2)



###########################################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()
#seaborn.set_context("talk")
seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})


[power1,kx,kt] = sp1
#[power2,kx,kt] = sp2
#[power3,kx,kt] = sp3
#power=np.log10(power1)-np.log10(power2)
#power=np.log10(power2)-np.log10(power3)
#power=np.log10(power3)
power=np.log10(power1)
print('calculated power')
print("power.shape ", power.shape)
print("min max power ", np.amin(power), np.amax(power))
#np.save(filename+"hp_power_tropics_m15_p15.npy",power)
#np.save(filename+"power_kx.npy",kx)
#np.save(filename+"power_kt.npy",kt)
#plt.figure(figsize=(8,8))
plt.figure(figsize=(5,5.5))
#contours = plt.contourf(kx,kt,power,levels=np.linspace(-4.4,4.4,20),cmap='RdYlBu_r') 
#contours = plt.contourf(kx,kt,power,levels=np.linspace(-1.2,2.2,20),cmap='RdYlBu_r',extend='both') 
contours = plt.contourf(kx,kt,power,levels=np.linspace(2.4,8.4,20),cmap='RdYlBu_r') 
plotDispCurves(Hbar)
plt.colorbar(contours,orientation='horizontal',shrink=0.95,aspect=30,pad=0.14,ticks=np.arange(-28.,28.1,2.0))
plt.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
plt.ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
plt.ylim((0,1))
#plt.ylim((0,1))
#plt.ylim((0,.5))
#plt.xlim((-75,75))
#plt.xlim((-15,15))
plt.xlim((-25,25))
plt.xlabel('Wavenumber $k_x$',size='12', fontname = 'Dejavu Sans')
plt.ylabel('Frequency [1/day]',size='12', fontname = 'Dejavu Sans')
plt.grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=100)
plt.legend(fontsize='10',loc='upper left')
plt.savefig('wk.png', bbox_inches='tight',dpi=150)
plt.close()
