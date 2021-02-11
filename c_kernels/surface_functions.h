// computing bulk formula for surface fluxes of u, v and T,
// following eq. (7) and (8) "Thatcher and Christiane Jablonowski, 2016"

#pragma once
#include <math.h>

void surface_bulk_formula(
           double g,
           double Rv,
           double Lv,
           double T_0,
           double Ch,
           double Cq,
           double Cd,
           double qv_star0,
           double eps_v,
           double* restrict p,
           double* restrict gz,
           double* restrict T,
           double* restrict qt,
           double* restrict T_surf,
           double* restrict u,
           double* restrict v,
           double* restrict u_surf_flux,
           double* restrict v_surf_flux,
           double* restrict T_surf_flux,
           double* restrict qt_surf_flux,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    double windspeed_za;
    double qt_surf;

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
            windspeed_za = g*sqrt(u[ijkmax-1]*u[ijkmax-1] + v[ijkmax-1]*v[ijkmax-1])/gz[ijkmax_p-1];
            qt_surf = qv_star0*eps_v/p[ijkmax_p]*exp(-Lv/Rv*(1.0/T_surf[ij] - 1.0/T_0));
            u_surf_flux[ij]  = -Cd*windspeed_za*u[ijkmax-1];
            v_surf_flux[ij]  = -Cd*windspeed_za*v[ijkmax-1];
            T_surf_flux[ij]  = -Ch*windspeed_za*(T[ijkmax-1] - T_surf[ij]);
            qt_surf_flux[ij] = -Cq*windspeed_za*(qt[ijkmax-1] - qt_surf);
        } // end j loop
    } // end i loop
    return;
}