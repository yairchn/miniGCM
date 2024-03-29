import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from math import *
from matplotlib.transforms import Transform
from matplotlib.ticker import (AutoLocator, AutoMinorLocator)

path='./res4/'
#path='./res1/'

ks=np.load(path+'ks.npy')
Ek=np.copy(ks)*0.
Ek_vrt=np.copy(ks)*0.
Ek_div=np.copy(ks)*0.
Ek_cross=np.copy(ks)*0.

#Layer=0
#Layer=1
#Layer=2

icount=0
for Layer in np.arange(0,3):
    for it in np.arange(420,800,14): Ek+=np.load(path+'Ek_flux_'+str(Layer)+'_0000000'+str(it)+'.npy'); icount+=1.
    for it in np.arange(420,800,14): Ek_vrt+=np.load(path+'EkRot_flux_'+str(Layer)+'_0000000'+str(it)+'.npy')
    for it in np.arange(420,800,14): Ek_div+=np.load(path+'EkDiv_flux_'+str(Layer)+'_0000000'+str(it)+'.npy')
    for it in np.arange(420,800,14): Ek_cross+=np.load(path+'EkCross_flux_'+str(Layer)+'_0000000'+str(it)+'.npy')


Ek/=icount
Ek_vrt/=icount
Ek_div/=icount
Ek_cross/=icount


fig, ax = plt.subplots(constrained_layout=True,figsize=(5,4.))

ax.semilogx(ks,Ek,'-k',linewidth=4,alpha=0.4,label='KE flux')
ax.semilogx(ks,Ek_vrt,'-r',label='KE rotational flux')
ax.semilogx(ks,Ek_div,'-b',label='KE divergent flux')
ax.semilogx(ks,Ek_cross,'--k',label='KE cross flux')
ax.semilogx(ks,Ek_vrt+Ek_div+Ek_cross,':r',label='KE overall flux')

ax.set_xlabel('Wavenumber $k$',size='12', fontname = 'Dejavu Sans')

ax.set_ylabel('Kinetic Energy Flux [W/kg]',size='12', fontname = 'Dejavu Sans')

plt.grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=100)

plt.xlim(1,700)

plt.ylim(-6.e-4,6.e-4)

plt.legend(loc='lower right')

circumference=6371.*pi*2. # [km]

def forward(x):
    return circumference / x 


def inverse(x):
    return circumference / x 

secax = ax.secondary_xaxis('top', functions=(forward, inverse))
secax.set_xlabel('Wavelength [km]',size='12', fontname = 'Dejavu Sans')
plt.tight_layout()
#plt.show()
plt.savefig('Kinetic_Energy_flux.pdf',dpi=150)

l=forward(ks)
f=1.e-4 # 1/s
#print('l [km]',l[1:170])

plt.clf()

exit()
