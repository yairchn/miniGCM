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
           double pv_star0,
           double eps_v,
           double* restrict p,
           double* restrict T,
           double* restrict qt,
           double* restrict ql,
           double* restrict T_mp,
           double* restrict qt_mp,
           double* restrict rain_rate,
           double* restrict qv_star,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double p_half;
    double denom;
    double Lv_Rv=Lv/Rv;
    double Lv_cpRv=pow(Lv,2.0)/cp/Rv;
    double Lv_cp=Lv/cp;
    double pv0epsv=pv_star0*eps_v;
    double T_0_inv=1.0/T_0;

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
                qv_star[ijk] = (pv0epsv/p_half)*exp(-Lv_Rv*(1.0/T[ijk]-T_0_inv)); // Eq. (1) Tatcher and Jabolonski 2016
                denom = (1.0+Lv_cpRv*qv_star[ijk]/(T[ijk]*T[ijk]))*dt;
                ql[ijk] = qt[ijk] - qv_star[ijk];
                T_mp[ijk]  =  (qt[ijk] - qv_star[ijk])/denom; // Eq. (2) Tatcher and Jabolonski 2016
                qt_mp[ijk] = -(qt[ijk] - (1.0+max_ss)*qv_star[ijk])/denom; // Eq. (3) Tatcher and Jabolonski 2016
                rain_rate[ij] = rain_rate[ij] - (qt_mp[ijk]/rho_w*g*(p[ijkmax_p]-p[0,0,0]))/(kmax); // Eq. (5) Tatcher and Jabolonski 2016
            } // end k loop
        } // end j loop
    } // end i loop
    return;
}