import scipy as sc
import numpy as np
from math import *

def eos(QT, T, Grid):

	p_tilde	   = Gr.p_ref
	Rd		   = Gr.Rd
	eps_vi	   = namelist['thermodynamics']['molar_mass_ratio']
	cpd		   = 1000.4
	kappa	   =  0.285956175299
	eps_v	   = 0.62210184182
	cpv		   = 1859.0
	ql		   = 0.0
	ql_2	   = 0.0
	qv_star_2  = 0.0


	pv_1 = p0 * eps_vi * qt /(1.0 - qt + eps_vi * qt)
	pd_1 = p0 - pv_1
	T_1 = T
	pv_star_1 = 6.1094*np.exp((17.625*(T_1 - 273.15))/((T_1 - 273.15)+243.04))*100.0
	qv_star_1 = eps_v * (1.0 - qt) * pv_star_1 / (p0 - pv_star_1)

	# If not saturated
	if(qt <= qv_star_1):
		T_out = T_1
		ql_out = 0.0

	else:
		ql_1 = qt - qv_star_1
		latent_heat = 2500.8-2.36*(T_1-273.15)+0.0016*(T_1-273.15)*(T_1-273.15)-0.00006*(T_1-273.15)*(T_1-273.15)*(T_1-273.15)*1000.0
		prog_1 = T_1 / ((p0/p_tilde)**kappa) * np.exp(-latent_heat*(ql_1/(1.0 - qt))/(T_1*cpd))
		f_1 = prog - prog_1
		T_2 = T_1 + ql_1 * latent_heat /((1.0 - qt)*cpd + qv_star_1 * cpv)
		delta_T  = np.abs(T_2 - T_1)

		while delta_T > 1.0e-6 or ql_2 < 0.0:
			pv_star_2 = 6.1094*np.exp((17.625*(T_2 - 273.15))/((T_2 - 273.15)+243.04))*100
			qv_star_2 = eps_v * (1.0 - qt) * pv_star_2 / (p0 - pv_star_2)
			pv_2      = p0 * eps_vi * qv_star_2 /(1.0 - qt + eps_vi * qv_star_2)
			pd_2      = p0 - pv_2
			ql_2      = qt - qv_star_2
			latent_heat = 2500.8-2.36*(T_2-273.15)+0.0016*(T_2-273.15)*(T_2-273.15)-0.00006*(T_2-273.15)*(T_2- 273.15) *(T_2-273.15)*1000.0
			prog_2    = T_2 / ((p0/p_tilde)**kappa) * np.exp(-latent_heat*(ql_2/(1.0 - qt))/(T_2*cpd))
			f_2       = prog - prog_2
			T_n       = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
			T_1       = T_2
			T_2       = T_n
			f_1       = f_2
			delta_T   = np.abs(T_2 - T_1)

		T_out  = T_2
		qv = qv_star_2
		ql_out = ql_2

	return T_out, ql_out