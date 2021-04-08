// A collection of tendency functions computed in a given k (vertical) level
// as some are conditioned upon a planner spcteral conversion in that level

#pragma once
#include <math.h>

void rhs_H(double* restrict H,
           double* restrict u,
           double* restrict v,
           double* restrict H_mp,
           double* restrict H_sur,
           double* restrict H_forc,
           double* restrict rhs_H,
           double* restrict u_H,
           double* restrict v_H,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax,
           ssize_t k)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            const ssize_t ijk = ishift + jshift + k;
            u_H[ij] = u[ijk] * H[ijk];
            v_H[ij] = v[ijk] * H[ijk];
            rhs_H[ij] = 0.0;
            rhs_H[ij] += H_mp[ijk];
            rhs_H[ij] += H_forc[ijk];
            rhs_H[ij] += H_sur[ij];
        } // End j loop
    } // end i loop
    return;
}


void rhs_qt(double* restrict qt,
            double* restrict u,
            double* restrict v,
            double* restrict qt_mp,
            double* restrict qt_sur,
            double* restrict rhs_qt,
            double* restrict u_qt,
            double* restrict v_qt,
            ssize_t imax,
            ssize_t jmax,
            ssize_t kmax,
            ssize_t k)
            {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    double dpi;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            const ssize_t ijk = ishift + jshift + k;
            rhs_qt[ij]  = 0.0;
            rhs_qt[ij] += qt_mp[ijk];
            rhs_qt[ij] += qt_sur[ij];
            u_qt[ij] = u[ijk] * qt[ijk];
            v_qt[ij] = v[ijk] * qt[ijk];
        } // End j loop
    } // end i loop
    return;
}


void RHS_momentum(double rho,
                  double* restrict p,
                  double* restrict vort,
                  double* restrict f,
                  double* restrict u,
                  double* restrict v,
                  double* restrict e_dry,
                  double* restrict u_vort,
                  double* restrict v_vort,
                  ssize_t imax,
                  ssize_t jmax,
                  ssize_t kmax,
                  ssize_t k)
                  {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            const ssize_t ijk = ishift + jshift + k;
            // dry energy should have the term that appears in the pressure gradient in SW
            e_dry[ij]  = p[ijk]/rho + 0.5*(u[ijk]*u[ijk]+v[ijk]*v[ijk]);
            u_vort[ij] = u[ijk] * (vort[ijk]+f[ij]);
            v_vort[ij] = v[ijk] * (vort[ijk]+f[ij]);
        } // end j loop
    } // end i loop
    return;
}
