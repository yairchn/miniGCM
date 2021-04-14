// computing the pointwise (in physical grid) turbulent flux following
// "Tatcher and Jablonowski, 2016"

#pragma once
#include <math.h>

void down_gradient_turbulent_flux(
           double c_e,
           double Dtheta_z,
           double T_equator,
           double DT_y,
           double* restrict p,
           double* restrict gz,
           double* restrict T,
           double* restrict qt,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double windspeed;
    double p_half;
    double K_e;
    double za;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift_2d = i*jmax;
        const ssize_t ishift = ishift_2d*kmax;
        const ssize_t ishift_p = ishift_2d*(kmax+1);
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t jshift_p = j*(kmax+1);
            const ssize_t ij = ishift_2d + j;
            const ssize_t ijkmax = ishift + jshift + kmax;
            const ssize_t ijkmax_p = ishift_p + jshift_p + kmax;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                const ssize_t ijkp = ishift_p + jshift_p + k;
                // eq. (17) Tatcher and Jablonowski 2016
                windspeed = sqrt(u[ijk]*u[ijk] + v[ijk]*v[ijk]);
                p_half = (p[ijkp]+p[ijkp+1])/2.0;
                z_a = gz[ijkmax_p-1]/g/2.0;
                K_e = c_e*windspeed*za*exp(-pow(max(Ppbl - p_half,0.0)/Pstrato),2.0)
                if (k==0){
                    Th_p = T[ijk+1]*pow(p_half/p_ref, kappa)
                    Th   = T[ijk]*pow(p_half/p_ref, kappa)
                    rhs_T[ij]  -= K_e*(T[ijk+1]  + T[ijk])*dpi;
                    rhs_qt[ij] -= K_e*(qt[ijk+1] + qt[ijk])*dpi;
                } // end if
                else if (k==kmax-1){
                    Th   = T[ijk]*pow(p_half/p_ref, kappa)
                    rhs_T[ijk]  -= K_e*(T[ijk]  + T[ijk-1])*dpi;
                    rhs_T[ijk]  -= K_e*(T[ijk]  + T[ijk])*dpi;
                    rhs_qt[ij] -= K_e*(qt[ijk] + qt[ijk-1])*dpi;
                    rhs_qt[ij] -= K_e*(qt[ijk] + qt[ijk])*dpi;
                } // end else if
                else{
                    Th_p = T[ijk+1]*pow(p_half/p_ref, kappa)
                    Th   = T[ijk]*pow(p_half/p_ref, kappa)
                    rhs_T[ij]  -= K_e*(T[ijk]    + T[ijk-1])*dpi;
                    rhs_T[ij]  -= K_e*(T[ijk+1]  + T[ijk])*dpi;
                    rhs_qt[ij] -= K_e*(qt[ijk]   + qt[ijk-1])*dpi;
                    rhs_qt[ij] -= K_e*(qt[ijk+1] + qt[ijk])*dpi;
                } // end else
            } // end k loop
        } // end j loop
    } // end i loop

    return;
}