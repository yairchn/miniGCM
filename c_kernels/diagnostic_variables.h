// Solving diagnostic functions for: pressure vertical velocity ω (Wp),
// geopotential ϕ (gZ) and kinetic energy (ke)
// at a given k (vertical) level as some are conditioned on a planner spcteral conversion
// Note - geopotential is computed diagnostically bottom -> up

#pragma once
#include <math.h>

void diagnostic_variables(
           double kappa,
           double Rd,
           double Rv,
           double Omega,
           double a,
           double* restrict lat,
           double* restrict p,
           double* restrict T,
           double* restrict qt,
           double* restrict ql,
           double* restrict u,
           double* restrict v,
           double* restrict div,
           double* restrict ke,
           double* restrict wp,
           double* restrict gz,
           double* restrict uv,
           double* restrict TT,
           double* restrict vT,
           double* restrict M,
           ssize_t k,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t k_rev = kmax-k-1;
    double Rm, exner;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift_2d = i*jmax;
        const ssize_t ishift = ishift_2d*kmax;
        const ssize_t ishift_p = ishift_2d*(kmax+1);
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t jshift_p = j*(kmax+1);
            const ssize_t ij = ishift_2d + j;
            const ssize_t ijk = ishift + jshift + k;
            const ssize_t ijkp = ishift_p + jshift_p + k;
            const ssize_t ijk_rev = ishift + jshift + k_rev;
            const ssize_t ijkp_rev = ishift_p + jshift_p + k_rev;
            ke[ijk]      = 0.5*(u[ijk]*u[ijk] + v[ijk]*v[ijk]);
            wp[ijkp+1]   = wp[ijkp] - (p[ijkp+1]-p[ijkp])*div[ijk];
            Rm           = Rd*(1.0-qt[ijk_rev]) + Rv*(qt[ijk_rev] - ql[ijk_rev]);
            //gz[ijkp_rev] = Rm*T[ijk_rev]*log(p[ijkp_rev+1]/p[ijkp_rev]) + gz[ijkp_rev+1];
            //gz[ijkp_rev] = Rm*T[ijk_rev]*(pow(p[ijkp_rev+1]/100000.,kappa)-pow(p[ijkp_rev]/100000.,kappa))/kappa + gz[ijkp_rev+1];
            //exner=pow(((p[ijkp_rev]+p[ijkp_rev+1])*0.5/100000.), kappa); 
            //gz[ijkp_rev] = Rm*exner*T[ijk_rev]*(p[ijkp_rev+1]-p[ijkp_rev])*2./(p[ijkp_rev]+p[ijkp_rev+1]) + gz[ijkp_rev+1];
            gz[ijkp_rev] = Rm*T[ijk_rev]*(p[ijkp_rev+1]-p[ijkp_rev])*2./(p[ijkp_rev]+p[ijkp_rev+1]) + gz[ijkp_rev+1];
            vT[ijk]      = v[ijk] * T[ijk];
            TT[ijk]      = T[ijk] * T[ijk];
            uv[ijk]      = v[ijk] * u[ijk];
            M[ijk]       = a*cos(lat[ij])*(a*Omega*cos(lat[ij]) + u[ijk]);
        } // end j loop
    } // end i loop
    return;
}
