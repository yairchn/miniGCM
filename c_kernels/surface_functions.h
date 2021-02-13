// computing bulk formula for surface fluxes of u, v and T,
// following eq. (7) and (8) "Thatcher and Christiane Jablonowski, 2016"

#pragma once
#include <math.h>

void surface_bulk_formula(
           double g,
           double Rv,
           double Lv,
           double T_0, ??
           double Ch,
           double Cq,
           double Cd,
           double qv_star0,
           double eps_v,
           double* restrict p,
           double* restrict h,
           double* restrict T,
           double* restrict qt,
           double* restrict u,
           double* restrict v,
           double* restrict u_surf_flux,
           double* restrict v_surf_flux,
           double* restrict qt_surf_flux,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    double z_a;
    double windspeed;
    double qt_surf;
    double Lv_Rv=Lv/Rv;
    double qv0epsv=qv_star0*eps_v;
    double T_0_inv=1.0/T_0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t ij = i*jmax + j;
            const ssize_t ijkmax = ishift + jshift + kmax;
            windspeed = sqrt(u[ijkmax-1]*u[ijkmax-1] + v[ijkmax-1]*v[ijkmax-1]);
            qt_surf = qv0epsv/p[ijkmax]*exp(-Lv_Rv*(1.0/T_surf[ij] - T_0_inv));
            z_a = h[ijkmax_p-1];
            u_surf_flux[ij]  = -Cd/z_a*windspeed*u[ijkmax-1];
            v_surf_flux[ij]  = -Cd/z_a*windspeed*v[ijkmax-1];
            qt_surf_flux[ij] = -Cq/z_a*windspeed*(qt[ijkmax-1] - qt_surf);
        } // end j loop
    } // end i loop
    return;
}