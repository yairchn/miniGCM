// computing a pointwise (in physical grid) cutoff microphysics
// and RainRate as vertical summation following eq. (1)-(5) in
// "Thatcher and Christiane Jablonowski, 2016"

#pragma once
#include <math.h>

void microphysics_cutoff(
           double cp,
           double dt,
           double Rv,
           double Lv,
           double T_0,
           double rho_w,
           double g,
           double max_ss,
           double qv_star0,
           double eps_v,
           double* restrict p,
           double* restrict T,
           double* restrict qt,
           double* restrict ql,
           double* restrict T_mp,
           double* restrict qt_mp,
           double* restrict rain_rate,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double p_half;
    double qv_star;
    double denom;
    double kmax_1 = kmax + 1.0;

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
            rain_rate[ij] = 0.0;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                const ssize_t ijkp = ishift_p + jshift_p + k;
                p_half = 0.5*(p[ijkp]+p[ijkp+1]);
                qv_star = (qv_star0*eps_v/p_half)*exp(-Lv/Rv*(1.0/T[ijk]-1.0/T_0));
                denom = (1.0+pow(Lv,2.0)/cp/Rv*qv_star/pow(T[ijk],2.0))*dt;
                ql[ijk] = fmax(qt[ijk] - qv_star,0.0);
                T_mp[ijk] = Lv/cp*ql[ijk]/denom;
                qt_mp[ijk] = -fmax((qt[ijk] - (1.0+max_ss)*qv_star), 0.0)/denom;
                rain_rate[ij] = rain_rate[ij] - (qt_mp[ijk]/rho_w*g*(p[ijkmax_p]-p[0,0,0]))/(kmax);
            } // end k loop
        } // end j loop
    } // end i loop
    return;
}