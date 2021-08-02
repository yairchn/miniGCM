import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from math import *
from matplotlib.transforms import Transform
from matplotlib.ticker import (AutoLocator, AutoMinorLocator)

Hbar=100.

ks=np.load('ks.npy')
Ek=np.copy(ks)*0.
Ek_vrt=np.copy(ks)*0.
Ek_div=np.copy(ks)*0.

#Layer=0
#Layer=1
#Layer=2

for Layer in np.arange(0,3):
    for it in np.arange(600,800,5): Ek+=np.load('Ek_flux_'+str(Layer)+'_0000000'+str(it)+'.npy')/40.
    for it in np.arange(600,800,5): Ek_vrt+=np.load('EkRot_flux_'+str(Layer)+'_0000000'+str(it)+'.npy')/40.
    for it in np.arange(600,800,5): Ek_div+=np.load('EkDiv_flux_'+str(Layer)+'_0000000'+str(it)+'.npy')/40.


#convert to eddy kinetic energy in shallow water system
Ek*=Hbar/2.
Ek_vrt*=Hbar/2.
Ek_div*=Hbar/2.

fig, ax = plt.subplots(constrained_layout=True,figsize=(5,4.))

ax.semilogx(ks[1:700],savgol_filter(Ek[1:700],5,1),'-k',linewidth=4,alpha=0.4,label='KE flux')
ax.semilogx(ks[1:700],savgol_filter(Ek_vrt[1:700],5,1),'-r',label='KE rotational flux')
ax.semilogx(ks[1:700],savgol_filter(Ek_div[1:700],5,1),'-b',label='KE divergent flux')

ax.set_xlabel('Wavenumber $k$',size='12', fontname = 'Dejavu Sans')

ax.set_ylabel('Eddy Kinetic Energy Flux [W/kg]',size='12', fontname = 'Dejavu Sans')

plt.grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=100)

plt.xlim(1,700)

#plt.ylim(1.e-3,10.)

plt.legend()

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