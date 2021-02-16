import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import AdamsBashforth
#import sphericalForcing as spf
import scipy as sc
import xarray
#import logData
#import netCDF4
#import matplotlib.ticker as mticker
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from numpy.fft import fft, ifft, fft2,ifft2,fftshift,fftfreq
#import spectra
import seaborn
#import dispersionCurves as dc
import waveFilter3D as wFilt
from scipy import signal
plt.ioff()

nlons  = 512  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)   # for gaussian grid.

# parameters for test
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5 # rotation rate
grav    = 9.80616 # gravity

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
longi,lati = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lati)
factor = np.cos(x.lats)

def calcFilt(name,tmin,tmax,dt,dx=1.):
    print (name)
    data = xarray.open_dataset(name)
    lats = data['latitude']
    lons = data['longitude']
    u    = data['uwind']
    v    = data['vwind']
    ht   = data['height']

    sel={}
    sel['latitude'] = slice(np.deg2rad(45),np.deg2rad(-45))
    sel['time']     = slice(tmin,tmax)
    usel   = u.loc[sel]
    vsel   = v.loc[sel]
    htsel  = ht.loc[sel]


    out  = htsel
    outu = usel
    lat  = out.latitude
    lon  = out.longitude
    time = out.time
    ntim = len(time)
    nlon = len(lon)
    nlat = len(lat)
    dx   = dx/nlon
    kxx  = fftfreq(nlon,dx)
    ktt  = fftfreq(ntim,dt)
    kyy  = np.arange(nlat)  # some dummy number
    [kt,ky,kx] = np.meshgrid(ktt,kyy,kxx,indexing='ij')
    kx   = fftshift(kx)
    kt   = fftshift(kt)

    htp  = htsel - htsel.mean(dim='longitude')
    up   = usel - usel.mean(dim='longitude')
    vp   = vsel - vsel.mean(dim='longitude')
    kmin1 = +1
    kmax1 = +50.
    kmin2 = -50.
    kmax2 = -1.

    frmin1 = 1e-3
    frmax1 = 0.5
    frmin2 = 1e-3
    frmax2 = 0.5

    out  = htp
    taper = signal.windows.hann(out.shape[0])
    out   = out*taper[:,np.newaxis,np.newaxis]
    outu = htp
    taper = signal.windows.hann(outu.shape[0])
    outu  = outu*taper[:,np.newaxis,np.newaxis]
    wf   = wFilt.waveFilter3D(out,np.deg2rad(10.),grav,rsphere,omega,dt=dt,dx=1.)
    htf1 = wf.user(kmin1,kmax1,frmin1,frmax1)
    htf2 = wf.user(kmin2,kmax2,frmin2,frmax2)

    wfu  = wFilt.waveFilter3D(outu,np.deg2rad(10.),grav,rsphere,omega,dt=dt,dx=1.)
    uf1 = wfu.user(kmin1,kmax1,frmin1,frmax1)
    uf2 = wfu.user(kmin2,kmax2,frmin2,frmax2)

    print("uf1 mean",np.mean(uf1))
    print("uf2 mean",np.mean(uf2))


#    out  = up
#    taper = signal.windows.hann(out.shape[0])
#    out   = out*taper[:,np.newaxis,np.newaxis]
#    wf   = wFilt.waveFilter3D(out,np.deg2rad(10.),grav,rsphere,omega,dt=dt,dx=1.)
#    uf1 = wf.user(kmin1,kmax1,frmin1,frmax1)
#    uf2 = wf.user(kmin2,kmax2,frmin2,frmax2)

#    out  = vp
#    taper = signal.windows.hann(out.shape[0])
#    out   = out*taper[:,np.newaxis,np.newaxis]
#    wf   = wFilt.waveFilter3D(out,np.deg2rad(10.),grav,rsphere,omega,dt=dt,dx=1.)
#    vf1 = wf.user(kmin1,kmax1,frmin1,frmax1)
#    vf2 = wf.user(kmin2,kmax2,frmin2,frmax2)

    data.close()

    return [htf1[100],htf2[100],np.rad2deg(lat),np.rad2deg(lon)]



def calcFiltHov(name,tmin,tmax,dt,dx=1.):
    print (name)
    data = xarray.open_dataset(name)
    lats = data['latitude']
    lons = data['longitude']
    u    = data['uwind']
    v    = data['vwind']
    ht   = data['height']

    sel={}
    sel['latitude'] = slice(np.deg2rad(25),np.deg2rad(-25))
    sel['time']     = slice(tmin,tmax,1)
    #usel   = u.loc[sel]
    #vsel   = v.loc[sel]
    htsel  = ht.loc[sel]


    out  = htsel-htsel.mean(dim='longitude')
    #outu = usel
    lat  = out.latitude
    lon  = out.longitude
    time = out.time
    ntim = len(time)
    nlon = len(lon)
    nlat = len(lat)
    dx   = dx/nlon
    kxx  = fftfreq(nlon,dx)
    ktt  = fftfreq(ntim,dt)
    kyy  = np.arange(nlat)  # some dummy number
    [kt,ky,kx] = np.meshgrid(ktt,kyy,kxx,indexing='ij')
    kx   = fftshift(kx)
    kt   = fftshift(kt)

    print("nlat, nlon, ntim", nlat, nlon, ntim)
    #exit()

    #up   = usel - usel.mean(dim='longitude')
    #vp   = vsel - vsel.mean(dim='longitude')
    kmin1 = +1
    kmax1 = +50.
    kmin2 = -50.
    kmax2 = -1.

    frmin1 = 1e-3
    frmax1 = 0.5
    frmin2 = 1e-3
    frmax2 = 0.5

    #taper = signal.windows.hann(out.shape[0])
    #out   = out*taper[:,np.newaxis,np.newaxis]
    #outu = htp
    #taper = signal.windows.hann(outu.shape[0])
    #outu  = outu*taper[:,np.newaxis,np.newaxis]
    wf   = wFilt.waveFilter3D(out,np.deg2rad(0.),grav,rsphere,omega,dt=dt,dx=1.)
    htf1 = wf.user(kmin1,kmax1,frmin1,frmax1)
    htf2 = wf.user(kmin2,kmax2,frmin2,frmax2)

    #wfu = wFilt.waveFilter3D(outu,np.deg2rad(10.),grav,rsphere,omega,dt=dt,dx=1.)
    #uf1 = wfu.user(kmin1,kmax1,frmin1,frmax1)
    #uf2 = wfu.user(kmin2,kmax2,frmin2,frmax2)

    #3d array: time, latitude, longitude
    #create a (hovmoeller)plot of time-longitude for latitude 0

    print("htf1.shape ", htf1.shape)    
    print("htf2.shape ", htf2.shape)    
    print("htf1 mean",np.mean(htf1))
    print("htf2 mean",np.mean(htf2))
 
    hov1=htf1[:,int(nlat/2),:]
    hov2=htf2[:,int(nlat/2),:]

    #print("uf1 mean",np.mean(uf1))
    #print("uf2 mean",np.mean(uf2))


#    out  = up
#    taper = signal.windows.hann(out.shape[0])
#    out   = out*taper[:,np.newaxis,np.newaxis]
#    wf   = wFilt.waveFilter3D(out,np.deg2rad(10.),grav,rsphere,omega,dt=dt,dx=1.)
#    uf1 = wf.user(kmin1,kmax1,frmin1,frmax1)
#    uf2 = wf.user(kmin2,kmax2,frmin2,frmax2)

#    out  = vp
#    taper = signal.windows.hann(out.shape[0])
#    out   = out*taper[:,np.newaxis,np.newaxis]
#    wf   = wFilt.waveFilter3D(out,np.deg2rad(10.),grav,rsphere,omega,dt=dt,dx=1.)
#    vf1 = wf.user(kmin1,kmax1,frmin1,frmax1)
#    vf2 = wf.user(kmin2,kmax2,frmin2,frmax2)

    data.close()

    return [hov1,hov2,np.rad2deg(lon),time]




########################################################

dt   = 0.1

path     = '/DATA/suhas/sw/'  # path to save the file
#filename = 'testEvap.nc'
#filename = 'testEvap_weakerForcing.nc'
#filename = 'testEvap_weakerForcing_asym.nc'
filename = 'testMoistVrtForcedWithEvapHeat.nc'

#varFilt  = calcFilt(path+filename,700,900,dt)

hovFilt  = calcFiltHov(path+filename,700,900,dt)


###############################################################

from mpl_toolkits.axes_grid1 import make_axes_locatable
from myCmap import *
from plotUtils import *
plt.ioff()
#seaborn.set_context("talk")
seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})
fig,ax = plt.subplots(1,2, sharey='all',figsize=(12,5),squeeze=False)
###################################
def plotFig(figName,varname,i):
    [ht1,ht2,lats,lons] = figName
    for j in range(2):
        ht = figName[j]
        plotLevels, plotTicks  = getLevelsAndTicks1(0.75*np.amax(abs(ht)),levels=16)
        contours = ax[i,j].contourf(lons,lats,ht,levels=plotLevels, cmap=joyDivCmapRdBl,extend='both',zorder=0)
        cb = plt.colorbar(contours,ax=ax[i,j], orientation='horizontal',shrink=0.8,aspect=30,pad=0.14)
        cb.set_ticks(plotTicks)
        cb.ax.set_xlabel('m',size='12', fontname = 'Dejavu Sans', labelpad=0.1)
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5],linewidth=1,zorder=10)

        ax[i,j].set_xticks(np.linspace(0,360,7))
        ax[i,j].set_yticks(np.linspace(-45,45,5)) 
        ax[i,j].set_ylim(-45,45)
        ax[i,j].set_xlabel('Longitude',size='12', fontname = 'Dejavu Sans')

        ax[0,0].set_ylabel('Latitude',size='12', fontname = 'Dejavu Sans')
        ax[0,0].set_title('Eastward',size='12', fontname = 'Dejavu Sans')
        ax[0,1].set_title('Westward ',size='12', fontname = 'Dejavu Sans')

        plt.savefig('/home/suhas/Desktop/filt'+varname, bbox_inches='tight')

def plotHov(figName,varname,i):
    [hov1,hov2,lons,time] = figName
    for j in range(2):
        hov = figName[j]
        plotLevels, plotTicks  = getLevelsAndTicks1(0.75*np.amax(abs(hov)),levels=16)
        contours = ax[i,j].contourf(lons,time,hov,levels=plotLevels, cmap=joyDivCmapRdBl,extend='both',zorder=0)
        cb = plt.colorbar(contours,ax=ax[i,j], orientation='horizontal',shrink=0.8,aspect=30,pad=0.14)
        cb.set_ticks(plotTicks)
        cb.ax.set_xlabel('Potential Energy [log m]',size='12', fontname = 'Dejavu Sans', labelpad=0.1)
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5],linewidth=1,zorder=10)

        ax[i,j].set_xticks(np.linspace(0,360,7))
        #ax[i,j].set_yticks(np.linspace(-45,45,5)) 
        #ax[i,j].set_ylim(-45,45)
        ax[i,j].set_xlabel('Longitude',size='12', fontname = 'Dejavu Sans')

        ax[0,0].set_ylabel('Time ',size='12', fontname = 'Dejavu Sans')
        ax[0,0].set_title('Eastward',size='12', fontname = 'Dejavu Sans')
        ax[0,1].set_title('Westward ',size='12', fontname = 'Dejavu Sans')

        plt.savefig('/home/suhas/Desktop/filt'+varname, bbox_inches='tight')

#######################
#plotFig(varFilt,'Height',0)

plotHov(hovFilt,'Height',0)





