// computing the pointwise (in physical grid) forcing following
// "Held Suarez, 1994"

#pragma once
#include <math.h>

void focring_hs(
           double k_a,
           double k_b,
           double k_f,
           double k_s,
           double Dtheta_z,
           double h_equator,
           double Dh_y,
           double* restrict p,
           double* restrict h,
           double* restrict h_bar,
           double* restrict sin_lat,
           double* restrict cos_lat,
           double* restrict u,
           double* restrict v,
           double* restrict u_forc,
           double* restrict v_forc,
           double* restrict h_forc,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double k_T;
    double k_v;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            const ssize_t ijkmax = ishift + jshift + kmax;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                h_bar[ijk] = (h_equator - Dh_y*sin_lat[ij]*sin_lat[ij] -
                                Dtheta_z*cos_lat[ij]*cos_lat[ij]);

                k_T = k_a + (k_s-k_a)*pow(cos_lat[ij],4.0);
                k_v = k_b +  k_f;

                u_forc[ijk] = -k_v *  u[ijk];
                v_forc[ijk] = -k_v *  v[ijk];
                h_forc[ijk] = -k_T * (h[ijk] - h_bar[ijk]);

            } // end k loop
        } // end j loop
    } // end i loop

    return;
}