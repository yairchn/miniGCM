// computing the pointwise (in physical grid) forcing following
// "Held Suarez, 1994"

#pragma once
#include <math.h>

void focring_bm(
           double kappa,
           double p_ref,
           double tau,
           double* restrict p,
           double* restrict T,
           double* restrict T_bar,
           double* restrict u,
           double* restrict v,
           double* restrict u_forc,
           double* restrict v_forc,
           double* restrict T_forc,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double p_half;
    double sigma_ratio;
    double k_T;
    double k_v;

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
                p_half = (p[ijkp]+p[ijkp+1])/2.0;
                // T_bar[ijk] = fmax(((T_equator - DT_y*sin_lat[ij]*sin_lat[ij] -
                //                 Dtheta_z*log(p_half/p_ref)*cos_lat[ij]*cos_lat[ij])*
                //                 pow(p_half/p_ref, kappa)),200.0);

                // k_T = k_a + (k_s-k_a)*sigma_ratio*pow(cos_lat[ij],4.0);
                // k_v = k_b +  k_f*sigma_ratio;

                // u_forc[ijk] = -k_v *  u[ijk];
                // v_forc[ijk] = -k_v *  v[ijk];
                // T_forc[ijk] = -k_T * (T[ijk] - T_bar[ijk]);

            } // end k loop
        } // end j loop
    } // end i loop

    return;
}