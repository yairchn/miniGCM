import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import os
import glob, os

def keSpectral_flux_error_dp(u,v,Geo,pU,pD,uD,vD,Wp):
    uk  = x.grdtospec(u)
    vk  = x.grdtospec(v)
    ukD = x.grdtospec(uD)
    vkD = x.grdtospec(vD)
    uak = x.grdtospec(Wp*u)
    vak = x.grdtospec(Wp*v)
    uakD= x.grdtospec(Wp*uD)
    vakD= x.grdtospec(Wp*vD)

    usp = 0.5*uak*uk.conj()
    vsp = 0.5*vak*vk.conj()
    uspD= 0.5*uakD*ukD.conj()
    vspD= 0.5*vakD*vkD.conj()
    Esp = uspD - usp + vspD - vsp

    # build spectrum
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    ks = np.arange(np.amax(l)+1)

    # running sum (Integral)
    for i in range(0,np.amax(l)+1):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
        #print('Ek['+str(i)+'] ', Ek[i])
        #exit()

    #print('l ',l)
    #exit()
    for i in range(0,np.amax(l)+1):
        for j in range(i,np.amax(l)+1):
            Ek_sum[i]+=Ek[j]
            #print('Ek_sum['+str(i)+'] j', Ek_sum[i], j)
    #print('Ek_sum[0] ', Ek_sum[0])
    #print('Ek_sum[9] ', Ek_sum[9])
    #exit()

    return [Ek_sum,ks]


def keSpectral_flux_dp(u,v,Geo,pU,pD):
    zero=np.copy(u)*0.

    # get vorticity, divergence in spectral space 
    vrtsp,divsp = x.getvrtdivspec(u,v)
    vrt=x.spectogrd(vrtsp)
    div=x.spectogrd(divsp)

    # pressure Up is pi+1
    # pressure Down pi-1
    # for pressure weighted ke flux
    dp=pU-pD

    k_b = 2.8935185185185185e-07
    uk      = x.grdtospec(u)
    vk      = x.grdtospec(v)
    dpk     = x.grdtospec(dp)
    dpx,dpy = x.getgrad(dpk) # get gradients in grid space
    psx,psy = x.getgrad(x.grdtospec(pU)) # get gradients in grid space
    ux, uy  = x.getgrad(uk) # get gradients in grid space
    vx, vy  = x.getgrad(vk) # get gradients in grid space
    Geok = x.grdtospec(Geo)
    Geox, Geoy = x.getgrad(Geok) # get gradients in grid space
    # (1)
    #uak = x.grdtospec(ux*u + uy*v)  # only non-linear divergence in flux
    #vak = x.grdtospec(vx*u + vy*v)
    # (2)
    #uak = x.grdtospec(ux*u*dp + uy*v*dp + div*u*dp + Geox*dp + k_b*u*dp)  # including the full flux
    #vak = x.grdtospec(vx*u*dp + vy*v*dp + div*v*dp + Geoy*dp + k_b*v*dp)
    # (3)
    #uak = x.grdtospec(dp*ux*u + dp*uy*v + dp*div*u)  # only non-linear divergence and divergent component in flux
    #vak = x.grdtospec(dp*vx*u + dp*vy*v + dp*div*v)
    # (4)
    uak = x.grdtospec(dp*ux*u + dp*uy*v + dp*div*u/2.)  # only non-linear divergence and divergent component in flux
    vak = x.grdtospec(dp*vx*u + dp*vy*v + dp*div*v/2.)
    # (6)
    #uak = x.grdtospec(dp*ux*u + dp*uy*v + dp*div*u/2. + dp*k_b*u)  # only non-linear divergence and divergent component in flux
    #vak = x.grdtospec(dp*vx*u + dp*vy*v + dp*div*v/2. + dp*k_b*v)

    Usp = -1.*uak*uk.conj()
    Vsp = -1.*vak*vk.conj()
    Esp = Usp + Vsp 

    # (5)
    pressure_contribution = x.grdtospec(u*dpx/2. + v*dpy/2.) 
    Esp-= pressure_contribution*uk*uk.conj() + pressure_contribution*vk*vk.conj() 

    # (7)
    momentum_error_x = x.grdtospec(0.5*Geo*psx) 
    momentum_error_y = x.grdtospec(0.5*Geo*psy) 
    Esp+= momentum_error_x*uk.conj() + momentum_error_y*vk.conj()


    # build spectrum
    Ek_sum = np.zeros(np.amax(l)+1)
    Ek = np.zeros(np.amax(l)+1)
    ks = np.arange(np.amax(l)+1)

    # running sum (Integral)
    for i in range(0,np.amax(l)+1):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])
        #print('Ek['+str(i)+'] ', Ek[i])
        #exit()

    #print('l ',l)
    #exit()
    for i in range(0,np.amax(l)+1):
        for j in range(i,np.amax(l)+1):
            Ek_sum[i]+=Ek[j]
            #print('Ek_sum['+str(i)+'] j', Ek_sum[i], j)
    #print('Ek_sum[0] ', Ek_sum[0])
    #print('Ek_sum[9] ', Ek_sum[9])
    #exit()

    return [Ek_sum,ks]



### Equations:
# u⃗⋆((u⃗·∇)u⃗) = (u⋆,v⋆)·(((u,v)·∇)(u,v)) = (u⋆,v⋆)·( (u∂/∂x,v∂/∂y)(u,v) ) =
# (u⋆,v⋆)·(u∂u/∂x + v∂u/∂y , u∂v/∂x + v∂v/∂y) =
# u⋆(u∂u/∂x + v∂u/∂y) + v⋆(u∂v/∂x + v∂v/∂y) = u⋆u∂u/∂x + u⋆v∂u/∂y + v⋆u∂v/∂x + v⋆v∂v/∂y =
# R{u∂/∂x[0.5uu⋆] + v∂/∂y[0.5uu⋆] + u∂/∂x[0.5vv⋆] + v∂/∂y[0.5vv⋆]} =
# 0.5 R{u∂/∂x[uu⋆+vv⋆] + v∂/∂y[uu⋆+vv⋆]}  = 0.5 R{(u⃗⋅∇)(u⃗⋅u⃗⋆)}

### plan:
# one loop of n layers and time
# load data in loop
# read from namelist or from stats file of a simulation the number of layer, time span
# make a new 'spectral_analysis' folder in the Output file of simulation where we store data and figures
# stitch to netDCF files.



# command line from main folder:
# python viz/contour_zonal_mean.py zonal_mean_T
def main():
    parser = argparse.ArgumentParser(prog='miniGCM')
    parser.add_argument("path")
    args = parser.parse_args()
    path = args.varname
    for file in glob.glob("*.in"):
        print(file)
        namelist_path = path + ''

	for Layer in np.arange(0,3):
		# load data
        print("day ", it," Layer ",Layer)
        ps=netCDF4.Dataset(path+'Pressure_'+str(it*3600*24)+'.nc').variables['Pressure'][:,:]
        u=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer]
        v=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer]
        T=netCDF4.Dataset(path+'Temperature_'+str(it*3600*24)+'.nc').variables['Temperature'][:,:,Layer]
        Geo=netCDF4.Dataset(path+'Geopotential_'+str(it*3600*24)+'.nc').variables['Geopotential'][:,:,Layer]
        print('Layer', Layer , 'Geo[0,0,Layer]',Geo[0,0]) 
        Wp=netCDF4.Dataset(path+'Wp_'+str(it*3600*24)+'.nc').variables['Wp'][:,:,Layer-1]/100.#hPa

        if Layer == 0:
            pD=ps*0.+p1
            pU=ps*0.+p2
            print('pU ',pU[1,1],' pD',pD[1,1],' Layer',Layer)
        elif Layer == 1:
            pD=ps*0.+p2
            pU=ps*0.+p3
            print('pU ',pU[1,1],' pD',pD[1,1],' Layer',Layer)
        elif Layer == 2:
            pD=ps*0.+p3
            pU=ps/100.
            print('pU ',pU[1,1],' pD',pD[1,1],' Layer',Layer)
        #pD=ps*0.+p1
        #pU=ps*0.+ps/100.
        print('pU ',pU[1,1],' pD',pD[1,1],' Layer',Layer)

		vrtsp,divsp = x.getvrtdivspec(u,v)
		u_vrt,v_vrt = x.getuv(vrtsp,divsp*0.)
		u_div,v_div = x.getuv(vrtsp*0.,divsp)
		ke_vrtsp,ke_divsp = x.getvrtdivspec((pD-pU)*u*0.5*(u*u+v*v),(pD-pU)*v*0.5*(u*u+v*v))
		ke_div = x.spectogrd(ke_divsp)
		if Layer==1 or Layer==2:
			uD=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer-1]
			vD=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer-1]
			print('Wp min ', np.amax(Wp), 'max', np.amin(Wp),' Layer',Layer)
			ke_error = Wp*(0.5*(uD*uD+vD*vD) - 0.5*(u*u+v*v)) 
		if Layer==2:
			Phi=netCDF4.Dataset(path+'Geopotential_'+str(it*3600*24)+'.nc').variables['Geopotential'][:,:,Layer]
			psx,psy=x.getgrad(x.grdtospec(ps/100.))
			total_energy_error = Phi*(u*psx+v*psy)/(pU-pD) 

		uD=netCDF4.Dataset(path+'U_'+str(it*3600*24)+'.nc').variables['U'][:,:,Layer-1]
		vD=netCDF4.Dataset(path+'V_'+str(it*3600*24)+'.nc').variables['V'][:,:,Layer-1]
		[KE_flux,ks] = keSpectral_flux_dp(u,v,Geo,pU,pD)
		[KE_flux_error,ks] = keSpectral_flux_error_dp(u,v,Geo,pU,pD,uD,vD,Wp)










if __name__ == '__main__':
    main()





