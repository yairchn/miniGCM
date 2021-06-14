import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os

# command line:
# python viz/profiles_t_one.py varname
def main():

    t1 = 280
    t2 = 623
    t_step = 7
    j=0
    latdeg0=np.load('./latdeg0.npy')[::4]

    i = 0
    suffix = ''
    #suffix = '_vrt'
    suffix = '_div'
    li=[4,3,2,1,.5]
    al=[0.4,0.6,0.8,0.9,1.]
    #la=['125 hPa','250 hPa','500 hPa','750 hPa','850 hPa']
    la=['250 hPa','500 hPa','750 hPa']
    #la=['250 hPa','350 hPa','450 hPa','550 hPa','650 hPa','750 hPa','850 hPa','950 hPa']
    fig,ax = plt.subplots(1,len(la), sharey='all',figsize=(5, 4),squeeze=False)
    for j in range(0,3):
        itcount=0
        var=0.*np.load('./ekin_zonalmean_'+str(j)+'_0000000'+str(t1)+'.npy')
        var2=0.*np.load('./ekin_zonalstd_'+str(j)+'_0000000'+str(t1)+'.npy')
        for it in range(t1,t2,t_step):
            var+=np.load('./ekin'+suffix+'_zonalmean_'+str(j)+'_0000000'+str(it)+'.npy')
            var2+=np.load('./ekin'+suffix+'_zonalstd_'+str(j)+'_0000000'+str(it)+'.npy')
            itcount+=1
        var/=float(itcount)
        var2/=float(itcount)

        ax[i,j].plot(var,latdeg0,'-k',linewidth=li[j],alpha=al[j],label=la[j])
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
        ax[i,j].set_yticks(np.linspace(-90,90,7))
        ax[i,j].set_ylim(-90,90)
        #ax[i,j].legend(loc='upper left')
    ax[0,0].set_ylabel('Latitude / $\circ$')
    #plt.xlim(-8,25)
    plt.grid(linestyle=':',alpha=0.6,linewidth=1)
    plt.suptitle("KE / m$^{2}$ s$^{-2}$",size='14', fontname = 'Dejavu Sans')
    plt.tight_layout()
    plt.savefig('ke'+suffix+'_3l.pdf',dpi=150)


    fig,ax = plt.subplots(1,len(la), sharey='all',figsize=(5, 4),squeeze=False)
    for j in range(0,3):
        itcount=0
        var=0.*np.load('./ekin_zonalmean_'+str(j)+'_0000000'+str(t1)+'.npy')
        var2=0.*np.load('./ekin_zonalstd_'+str(j)+'_0000000'+str(t1)+'.npy')
        for it in range(t1,t2,t_step):
            var+=np.load('./ekin'+suffix+'_zonalmean_'+str(j)+'_0000000'+str(it)+'.npy')
            var2+=np.load('./ekin'+suffix+'_zonalstd_'+str(j)+'_0000000'+str(it)+'.npy')
            itcount+=1
        var/=float(itcount)
        var2/=float(itcount)

        ax[i,j].plot(var2,latdeg0,'-k',linewidth=li[j]/2.,alpha=al[j],label=la[j])
        ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
        ax[i,j].set_yticks(np.linspace(-90,90,7))
        ax[i,j].set_ylim(-90,90)
        #ax[i,j].legend(loc='upper left')
    ax[0,0].set_ylabel('Latitude / $\circ$')
    #plt.xlim(-8,25)
    plt.grid(linestyle=':',alpha=0.6,linewidth=1)
    plt.suptitle("EKE / m$^{2}$ s$^{-2}$",size='14', fontname = 'Dejavu Sans')
    plt.tight_layout()
    plt.savefig('eke'+suffix+'_3l.pdf',dpi=150)


if __name__ == '__main__':
    main()
