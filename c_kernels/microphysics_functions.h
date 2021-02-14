// computing a pointwise (in physical grid) cutoff microphysics
// and RainRate as vertical summation following eq. (1)-(5) in
// "Thatcher and Christiane Jablonowski, 2016"

#pragma once
#include <math.h>

void microphysics_cutoff(
           double dt,
           double rho_w,
           double g,
           double max_ss,
           double* restrict p,
           double* restrict h,
           double* restrict qt,
           double* restrict ql,
           double* restrict h_mp,
           double* restrict qt_mp,
           double* restrict rain_rate,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double qv_star;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            const ssize_t ijkmax = ishift + jshift + kmax;
            rain_rate[ij] = 0.0;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                qv_star = 0.001;
                ql[ijk] = fmax(qt[ijk] - qv_star,0.0);
                h_mp[ijk] = ql[ijk];
                qt_mp[ijk] = -fmax((qt[ijk] - (1.0+max_ss)*qv_star), 0.0);
                rain_rate[ij] = rain_rate[ij] - (qt_mp[ijk]/rho_w*g*(p[ijk]-p[0,0,0]))/(kmax);
            } // end k loop
        } // end j loop
    } // end i loop
    return;
}