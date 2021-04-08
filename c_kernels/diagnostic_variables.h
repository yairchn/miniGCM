// Solving diagnostic functions for: pressure vertical velocity ω (Wp),
// geopotential ϕ (gZ) and kinetic energy (ke)
// at a given k (vertical) level as some are conditioned on a planner spcteral conversion
// Note - geopotential is computed diagnostically bottom -> up

#pragma once
#include <math.h>

void diagnostic_variables(
           double g,
           double Omega,
           double a,
           double* restrict lat,
           double* restrict rho,
           double* restrict H_tot,
           double* restrict p,
           double* restrict h,
           double* restrict qt,
           double* restrict ql,
           double* restrict u,
           double* restrict v,
           double* restrict div,
           double* restrict ke,
           double* restrict uv,
           double* restrict hh,
           double* restrict vh,
           double* restrict M,
           ssize_t k,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            const ssize_t ijk = ishift + jshift + k;
            ke[ijk]       = 0.5*(u[ijk]*u[ijk] + v[ijk]*v[ijk]);
            vh[ijk]       = v[ijk]*h[ijk];
            hh[ijk]       = h[ijk]*h[ijk];
            uv[ijk]       = v[ijk]*u[ijk];
            M[ijk]        = a*cos(lat[ij])*(a*Omega*cos(lat[ij]) + u[ijk]);
            if (k==0){
                p[ijk] =  rho[k]*g*H_tot[i,j,k];
            } // end if
            else{
                p[ijk] =  p[ijk-1] + rho[k]*g*H_tot[i,j,k];
            } // end else
        } // end j loop
    } // end i loop
    return;
}

void total_depth(
           double* restrict H_tot,
           double* restrict h,
           ssize_t kk,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                const ssize_t ijkk = ishift + jshift + kk;
                H_tot[ijkk] += h[ijk];
            } // end k loop
        } // end j loop
    } // end i loop
    return;
}