import numpy as np
import sphTrans

class sphForcing(object):
    def __init__(self, nlons, nlats, ntrunc, rsphere, lmin, lmax, magnitude , correlation = 0.5, noise_type = 'white'):
        self.lmin = lmin
        self.lmax = lmax
        self.corr = correlation
        self.magnitude = magnitude
        self.ntrunc = ntrunc
        self.rsphere = rsphere
        self.trans = sphTrans.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
        self.l = self.trans._shtns.l
        self.m = self.trans._shtns.m
        self.noise_type = noise_type
        A = np.zeros(self.trans.nlm)
        A[self.l >= self.lmin] = 1.
        A[self.l >  self.lmax] = 0.
        A[self.m == 0] = 0.     # Removing zonal mean
        self.A = A
        self.nlm = self.trans._shtns.nlm

    def forcingFn(self,F0):
        wavenumber=self.l.astype(float); wavenumber[0]=1 #just to prepare for scaling
        if self.noise_type == 'white':
            signal = self.magnitude*self.A*wavenumber**(-0.5)* np.exp(np.random.rand(self.nlm)*1j*2*np.pi) # white noise
        elif self.noise_type == 'blue':
            signal = self.magnitude* self.A*wavenumber**(-0.25)* np.exp(np.random.rand(self.nlm)*1j*2*np.pi) # blue noise
        elif self.noise_type == 'red':
            signal = self.magnitude* self.A*wavenumber**(-0.8)* np.exp(np.random.rand(self.nlm)*1j*2*np.pi) # red noise
        elif self.noise_type == 'local':
            signal = self.magnitude* self.A *np.exp(np.random.rand(self.nlm)*1j*2*np.pi)
        F = (np.sqrt(1-self.corr**2))*signal + self.corr*F0
        out = F
        out[self.m==0] = 0. # Remove zonal mean component
        return out
