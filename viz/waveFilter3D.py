#####################################################################
# Function to filter signal outside a given wave dispersion regions
# Suhas D L   21/11/2017
#  Input the 3D signal
# Output is the filtered signal
# Signal(time,lat,lon)
#####################################################################

import numpy as np
import xarray
from numpy.fft import fft,fft2,ifft2,fftshift,ifftshift,fftfreq

class waveFilter3D(object):

    def __init__(self,var,lat,grav,rsphere,omega,dt,dx=1.,mean=0):
        self.ll    = 2*np.pi*rsphere*np.cos(lat)
        self.beta  = 2*omega*np.cos(lat)/rsphere
        self.g     = grav
        self.mean  = mean
        self.dx    = dx
        self.dt    = dt
        [ntim,nlat,nlon] = var.shape
        dx         = dx/nlon
        kxx        = fftfreq(nlon,dx)
        ktt        = fftfreq(ntim,dt)
        kyy        = np.arange(nlat)  # some dummy number
        [kt,ky,kx] = np.meshgrid(ktt,kyy,kxx,indexing='ij')
        self.kx    = fftshift(kx,axes=(0,2))
        self.kt    = fftshift(kt,axes=(0,2))
        self.kdim  = self.kx * (2*np.pi/self.ll)
        self.varfft  = self.Fourier(var)

    def Fourier(self,var):
        tmp   = fft2(var,axes=(0,2))
        out   = fftshift(tmp,axes=(0,2))
        out   = np.flip(out,axis=2)
        return out

    def invFourier(self,var):
        tmp   = np.flip(var,axis=2)
        tmp   = ifftshift(tmp,axes=(0,2))
        out   = ifft2(tmp,axes=(0,2))
        return out.real


    def MRGF(self,HMin,HMax,kMin,kMax):
        cMin   = np.sqrt(self.g * HMin)
        cMax   = np.sqrt(self.g * HMax)
        sigma1 = np.zeros_like(self.kdim)
        sigma2 = np.zeros_like(self.kdim)
        de1    = np.sqrt( 1 + (4*self.beta)/((self.kdim**2)* cMin))
        de2    = np.sqrt( 1 + (4*self.beta)/((self.kdim**2)* cMax))

        sigma1[self.kdim<0]  = self.kdim[self.kdim<0] * cMin * (0.5-0.5*de1[self.kdim<0])
        sigma1[self.kdim==0] = np.sqrt(cMin*self.beta)
        sigma1[self.kdim>0]  = self.kdim[self.kdim>0] * cMin * (0.5+0.5*de1[self.kdim>0])
        sigma1 = sigma1 + self.mean* self.kdim

        sigma2[self.kdim<0]  = self.kdim[self.kdim<0] * cMax * (0.5-0.5*de2[self.kdim<0])
        sigma2[self.kdim==0] = np.sqrt(cMax*self.beta)
        sigma2[self.kdim>0]  = self.kdim[self.kdim>0] * cMax * (0.5+0.5*de2[self.kdim>0])
        sigma2 = sigma2 + self.mean* self.kdim

        sigMax = np.maximum.reduce([sigma1,sigma2])
        sigMin = np.minimum.reduce([sigma1,sigma2])
        fMin   = (sigMin*24.*60.*60.) / (2*np.pi)
        fMax   = (sigMax*24.*60.*60.) / (2*np.pi)
        win    = np.zeros_like(self.varfft)
        win[(self.kx>=kMin) & (self.kx<=kMax) & (self.kt>=fMin) & (self.kt<=fMax)] = 1.
        # win    = np.ones_like(self.varfft)
        tmp    = self.varfft * win
        out    = self.invFourier(tmp)

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure()
        # plt.contourf(self.kx[:,20,:],self.kt[:,20,:],abs(win[:,20,:]),16,cmap='coolwarm')
        # plt.colorbar()
        # plt.xlim(-10,10)
        # plt.ylim(0,2)

        return out
##########################################################################
    def user(self,kMin,kMax,fMin,fMax):
        win    = np.zeros_like(self.varfft)
        # win[(self.kx>=kMin) & (self.kx<=kMax) & (self.kt>=fMin) & (self.kt<=fMax)] = 1.
        # win[(self.kx<=-kMin) & (self.kx>=-kMax) & (self.kt<=-fMin) & (self.kt>=-fMax)] = 1.
        win[(self.kx>=kMin-1) & (self.kx<=kMax-1) & (self.kt>=fMin) & (self.kt<=fMax)] = 1.
        win[(self.kx<=-kMin-1) & (self.kx>=-kMax-1) & (self.kt<=-fMin) & (self.kt>=-fMax)] = 1.
        tmp    = self.varfft * win
        out    = self.invFourier(tmp)
        return out

###########################################################################
    # def east(self,kMax=10):
    #     win    = np.zeros_like(self.varfft)
    #     win[(abs(self.kx)>=1) & (abs(self.kx)<=kMax) & (self.kt>0)] = 1.
    #     tmp    = self.varfft * win
    #     out    = self.invFourier(tmp)
    #     return out
