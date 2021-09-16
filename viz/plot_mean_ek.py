#Code to solve moist shallow water equations on a sphere
#SWE are of the vorticity-divergence form.
#
# run to sum over temporal snapshots of spectra of plot_ek.py
#

import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from scipy.signal import savgol_filter


path='./res1/'
path='./res4/'
path='./'

ks=np.load(path+'ks.npy')

Layer=0
it=420
EkTot =np.load(path+'EkTot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')*0.
EkRot =np.load(path+'EkRot_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')*0.
EkDiv =np.load(path+'EkDiv_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')*0.
#EkMean=np.load(path+'EkMean_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')*0.

icount=0
for Layer in np.arange(0,1):
    print("Layer ",Layer)
    for it in np.arange(420,630,14):
        print('it ',it)
        icount+=1
        #
        # Energy Spectra
        #
        #
        EkTot+=np.load(path+'EkTot_' +str(Layer)+'_'+str(it).zfill(10)+'.npy')
        EkRot+=np.load(path+'EkRot_' +str(Layer)+'_'+str(it).zfill(10)+'.npy')
        EkDiv+=np.load(path+'EkDiv_' +str(Layer)+'_'+str(it).zfill(10)+'.npy')
        #EkMean+=np.load(path+'EkMean_'+str(Layer)+'_'+str(it).zfill(10)+'.npy')
EkTot/=icount
EkRot/=icount
EkDiv/=icount
#EkMean/=icount

plt.figure(figsize=(5.5,4.5))
plt.clf()
EkTot[0]=EkTot[1]
EkRot[0]=EkRot[1]
EkDiv[0]=EkDiv[1]
EkTot=savgol_filter(EkTot,5,1)
EkRot=savgol_filter(EkRot,5,1)
EkDiv=savgol_filter(EkDiv,5,1)
plt.loglog(ks,EkTot,'-',color='black',alpha=0.4,linewidth=4,label='KE')
plt.loglog(ks,EkRot,'-',color='red',label='KE vortical')
plt.loglog(ks,EkDiv,'-',color='blue',label='KE divergent')
plt.loglog(ks[10:100],5.e1*ks[10:100]**(-5./3.),'--k',linewidth=2,label='-5/3')
plt.title('KE Spectra / m$^2$ s$^{-2}$')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('Wavenumber')
#plt.ylim(1.e-3,1.e5)
plt.ylim(5.e-3,5.e2)
plt.xlim(1.,2.e2)
plt.tight_layout()
#plt.savefig(path+'Ek.pdf')
plt.savefig('Ek.pdf')



    #plt.figure(figsize=(5.5,4.))
    #plt.clf()
    #EkTot[1:170]=savgol_filter(EkTot[1:170]/float(icount),5,1)
    #EkRot[1:170]=savgol_filter(EkRot[1:170]/float(icount),5,1)
    #EkDiv[1:170]=savgol_filter(EkDiv[1:170]/float(icount),5,1)
    #plt.loglog(ks,EkTot,'-',color='black',alpha=0.4,linewidth=4,label='KE')
    #plt.loglog(ks,EkRot,'-',color='red',label='KE vortical')
    #plt.loglog(ks,EkDiv,'-',color='blue',label='KE divergent')
    #plt.loglog(ks,8.e5*ks**(-3.),'-k',linewidth=2,label='-3')
    #plt.loglog(ks[15:400],5.e1*ks[15:400]**(-5./3.),'--k',linewidth=2,label='-5/3')
    #plt.title('KE Spectra / m$^2$ s$^{-2}$')
    #if Layer==0: plt.legend(loc='upper left')
    #plt.grid()
    #plt.ylim(1.e-3,1.e3)
    #plt.xlim(1.,2.e2)
    #plt.savefig('Ek_'+str(Layer)+'_'+str(it).zfill(10)+'.pdf')


