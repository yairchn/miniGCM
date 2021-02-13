// Solving diagnostic functions for: pressure vertical velocity ω (Wp),
// geopotential ϕ (gZ) and kinetic energy (ke)
// at a given k (vertical) level as some are conditioned on a planner spcteral conversion
// Note - geopotential is computed diagnostically bottom -> up

#pragma once
#include <math.h>

void diagnostic_variables(
           double Rd,
           double Rv,
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
           ssize_t k,
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
            const ssize_t ijk = ishift + jshift + k;
            ke[ijk]      = 0.5*(u[ijk]*u[ijk] + v[ijk]*v[ijk]);
            p[ijk]       = rho[k]*g*h[ijk];
            vh[ijk]      = v[ijk]*h[ijk];
            hh[ijk]      = h[ijk]*h[ijk];
            uv[ijk]      = v[ijk]*u[ijk];
        } // end j loop
    } // end i loop
    return;
}