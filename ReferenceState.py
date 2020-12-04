import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sc
from math import *

class ReferenceState:
	def __init__(self):
		# self.p = np.zeros(Pr.n_layers, dtype=np.double, order='c')
		return

	def initialize(self, Gr):
		self.Tbar = np.zeros((Pr.nlats,Pr.nlons, Pr.n_layers), dtype=np.double, order='c')
		self.Base_pressure = np.zeros((Pr.nlats,Pr.nlons, Pr.n_layers+1), dtype=np.double, order='c')
		# need to initlize
		# coordinates in meridional direction for pressure level p
		y=Gr.lat[:,0]
		print("y.shape: ",y.shape)
		y_deg=y*360./(2.*pi)

		# calculate damping coefficients in sigma coordinates

		s=np.arange(0.,1.,1.e-2)
		sigma_ratio=(s-self.sigma_b)/(1-self.sigma_b)
		sigma_ratio[sigma_ratio<0.]=0.

		k_v=k_f*sigma_ratio

		nz =s.shape[0]
		ny =y.shape[0]
		k_T=np.zeros((ny,nz))
		print("shape of k_v, k_T:",k_v.shape,k_T.shape)
		for jj in np.arange(0,ny,1):
		    k_T[jj,:]=k_a+(k_s-k_a)*sigma_ratio*np.cos(y[jj])**4
		    #print("min max k_T["+str(jj).strip()+",:]",np.amin(k_T[jj,:]),np.amax(k_T[jj,:]))

		#Temperature profiles from Held and Suarez (1994) -- initialize isothermally in each layer
		Tbar1_const   = 229.      # mean temp (some typical range) [K]
		Tbar2_const   = 257.      # mean temp (some typical range) [K]
		Tbar3_const   = 295.      # mean temp (some typical range) [K]
		#
		Tbar1_const*=cp
		Tbar2_const*=cp
		Tbar3_const*=cp

		#Tamp =  5.  # amplitude of forcing [K]
		Tamp = 25.  # amplitude of forcing [K]
		#Tamp =  0.  # amplitude of forcing [K]
		#S    = cp*Tamp * np.exp(-((lons-lons_ref)/(delta_lons))**2) * np.exp(-((lats-lats_ref)/(delta_lats))**2)
		#print("Temperature forcing min max:",np.amin(S),np.amax(S))

		# Relaxation time scales
		# May be increase tau_s  to 1000 or 2000 days to see the formation of jets
		# But make sure to run the simulation till at least twice this value
		tau_vrt  = 50*24*3600.
		tau_div  = 50*24*3600.
		tau_temp = 50*24*3600.


		tau_q   = 10*24*3600.   # evaporation time scale
		tau_c   = 0.1*24*3600.  # condensation time scale

		chi = 5500.  #### coversion factor similar to latent heat
		# make sure that (h - chi*Qsat) is always positive
		######## Qsat profile ##################
		lats_refQ   = np.radians(0)
		lons_refQ   = np.radians(180)  # change here
		delta_latsQ = np.radians(60)
		delta_lonsQ = np.radians(120)
		Qbar_amp3= 50e-3
		Qbar_amp2= 50e-3
		Qbar_amp1= 50e-3
		# gaussian in lat, lon , lat & lon and constant profiles.
		#Qbar     = Qbar_amp * np.exp(-((lats-lats_refQ)/(delta_latsQ))**2)*np.exp(-((lons-np.pi)/(delta_lonsQ))**2)
		#Qbar     = Qbar_amp * np.exp(-((lons-np.pi)/(delta_lonsQ))**2)
		#Qbar     = Qbar_amp * np.exp(-((lats-lats_refQ)/(delta_latsQ))**2)
		Qbar3    = Qbar_amp3* np.ones_like(lats)  # constant Qsat
		Qbar2    = Qbar_amp2* np.ones_like(lats)  # constant Qsat
		Qbar1    = Qbar_amp1* np.ones_like(lats)  # constant Qsat


		return 



	def update():
	    S    = cp*Tamp * np.exp(-((lons-lons_ref)/(delta_lons))**2) * np.exp(-((lats-lats_ref)/(delta_lats))**2) # maybe not needed now 

	    #Temperature profiles from Held and Suarez (1994)
	    Tbar1=np.copy(zero_array)
	    Tbar2=np.copy(zero_array)
	    Tbar3=np.copy(zero_array)
	    for jj in np.arange(0,nlons,1):
	        Tbar1[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p1/ps[:,jj])*np.cos(y)**2)*(p1/ps[:,jj])**kappa
	        Tbar2[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p2/ps[:,jj])*np.cos(y)**2)*(p2/ps[:,jj])**kappa
	        Tbar3[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p3/ps[:,jj])*np.cos(y)**2)*(p3/ps[:,jj])**kappa
	        #Tbar1[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p1/p0)*np.cos(y)**2)*(p1/p0)**kappa
	        #Tbar2[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p2/p0)*np.cos(y)**2)*(p2/p0)**kappa
	        #Tbar3[:,jj]=(315.-DT_y*np.sin(y)**2-Dtheta_z**2*np.log(p3/p0)*np.cos(y)**2)*(p3/p0)**kappa
	    Tbar1[Tbar1<=200.]=200. # minimum equilibrium Temperature is 200 K
	    Tbar2[Tbar2<=200.]=200. # minimum equilibrium Temperature is 200 K
	    Tbar3[Tbar3<=200.]=200. # minimum equilibrium Temperature is 200 K
	    Tbar1*=cp
	    Tbar2*=cp
	    Tbar3*=cp
	    #Tbar1+=Tbar1_const
	    #Tbar2+=Tbar2_const
	    #Tbar3+=Tbar3_const

	    return 

	 # def PlotReference():
	 # 	#
		# plt.figure(figsize=(4,4))
		# plt.clf()
		# plt.plot(k_v[::-1],s[::-1],'--k',linewidth=2)
		# plt.ylim((1.,0.))
		# plt.xlim((-.1,1.2))
		# plt.xlabel('$k_v$ [1/day]')
		# plt.ylabel('$\sigma$')
		# plt.title('Rayleigh damping $k_v$')
		# plt.tight_layout()
		# #plt.savefig('hs_kv.png')
		# #plt.close()
		# plt.pause(1e-3)
		# #
		# plt.figure(figsize=(4,4))
		# plt.clf()
		# out=np.transpose(k_T-0.0250)[::-1,:]
		# out=np.transpose(k_T)[::-1,:]
		# dlevels=np.amax(out)-np.amin(out)
		# cs=plt.contourf(y_deg,s,out,levels=np.arange(np.amin(out),np.amax(out),dlevels/100.))
		# plt.colorbar(cs,ticks=np.arange(k_a,k_s,k_a))
		# plt.xlabel('Meridional direction / $\circ$')
		# plt.ylabel('$\sigma$')
		# plt.title('Rayleigh damping $k_T$ [1/day]')
		# plt.tight_layout()
		# #plt.savefig('hs_kT.png')
		# #plt.close()
		# plt.pause(1e-3)
		# return 

