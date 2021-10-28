// A collection of tendency functions computed in a given k (vertical) level
// as some are conditioned upon a planner spcteral conversion in that level

#pragma once
#include <math.h>

void rhs_E(double cp,
           double* restrict p,
           double* restrict gz,
           double* restrict E,
           double* restrict u,
           double* restrict v,
           double* restrict wp,
           double* restrict E_mp,
           double* restrict E_sur,
           double* restrict E_forc,
           double* restrict turbflux,
           double* restrict rhs_E,
           double* restrict u_E_gz,
           double* restrict v_E_gz,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax,
           ssize_t k)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    double dpi;

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
            u_E_gz[ij] = u[ijk] * (E[ijk] + gz[ijk]);
            v_E_gz[ij] = v[ijk] * (E[ijk] + gz[ijk]);
            dpi = 1.0/(p[ijkp+1] - p[ijkp]);
            rhs_E[ij] = 0.0;
            if (k==0){
                rhs_E[ij] -= 0.5*wp[ijkp+1]*(E[ijk+1] + E[ijk])*dpi;
                rhs_E[ij] -= (turbflux[ijk+1] - turbflux[ijk])*dpi;
            } // end if
            else if (k==kmax-1){
                rhs_E[ij] += 0.5*wp[ijkp]  *(E[ijk] + E[ijk-1])*dpi;
                rhs_E[ij] -= 0.5*wp[ijkp+1]*(E[ijk] + E[ijk])*dpi;
                rhs_E[ij] -= (E_sur[ij] - turbflux[ijk])*dpi;
            } // end else if
            else{
                rhs_E[ij] += 0.5*wp[ijkp]  *(E[ijk]   + E[ijk-1])*dpi;
                rhs_E[ij] -= 0.5*wp[ijkp+1]*(E[ijk+1] + E[ijk])*dpi;
                rhs_E[ij] -= (turbflux[ijk+1] - turbflux[ijk])*dpi;
            } // end else
            rhs_E[ij] -= 0.5*(wp[ijkp+1]+wp[ijkp])*(gz[ijkp+1]-gz[ijkp])*dpi/cp;
            rhs_E[ij] += E_mp[ijk];
            rhs_E[ij] += E_forc[ijk];
        } // End j loop
    } // end i loop
    return;
}




void rhs_qt(double* restrict p,
            double* restrict qt,
            double* restrict u,
            double* restrict v,
            double* restrict wp,
            double* restrict qt_mp,
            double* restrict qt_sur,
            double* restrict turbflux,
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
        const ssize_t ishift_2d = i*jmax;
        const ssize_t ishift = ishift_2d*kmax;
        const ssize_t ishift_p = ishift_2d*(kmax+1);
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*kmax;
            const ssize_t jshift_p = j*(kmax+1);
            const ssize_t ij = ishift_2d + j;
            const ssize_t ijk = ishift + jshift + k;
            const ssize_t ijkp = ishift_p + jshift_p + k;
            rhs_qt[ij]  = 0.0;
            dpi = 1.0/(p[ijkp+1] - p[ijkp]);
            if (k==0){
                rhs_qt[ij] -= 0.5*wp[ijkp+1]*(qt[ijk+1]+ qt[ijk])*dpi;
                rhs_qt[ij] -= (turbflux[ijk+1] - turbflux[ijk])*dpi;
            } // end if
            else if (k==kmax-1){
                rhs_qt[ij] += 0.5*wp[ijkp]  *(qt[ijk] +qt[ijk-1])*dpi;
                rhs_qt[ij] -= 0.5*wp[ijkp+1]*(qt[ijk] +qt[ijk])*dpi;
                rhs_qt[ij] -= (qt_sur[ij] - turbflux[ijk])*dpi;
            } // end else if
            else{
                rhs_qt[ij] += 0.5*wp[ijkp]  *(qt[ijk]   + qt[ijk-1])*dpi;
                rhs_qt[ij] -= 0.5*wp[ijkp+1]*(qt[ijk+1] + qt[ijk])*dpi;
                rhs_qt[ij] -= (turbflux[ijk+1] - turbflux[ijk])*dpi;
            } // end else
            rhs_qt[ij] += qt_mp[ijk];
            u_qt[ij] = u[ijk] * qt[ijk];
            v_qt[ij] = v[ijk] * qt[ijk];
        } // End j loop
    } // end i loop
    return;
}


void vertical_uv_fluxes(double* restrict p,
                        double* restrict gz,
                        double* restrict vort,
                        double* restrict f,
                        double* restrict u,
                        double* restrict v,
                        double* restrict wp,
                        double* restrict ke,
                        double* restrict wdudp_up,
                        double* restrict wdvdp_up,
                        double* restrict wdudp_dn,
                        double* restrict wdvdp_dn,
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
    double dpi;

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
            dpi = 1.0/(p[ijkp+1] - p[ijkp]);
            if (k==0){
                wdudp_dn[ij] = 0.5*wp[ijkp+1]*(u[ijk+1] - u[ijk])*dpi;
                wdvdp_dn[ij] = 0.5*wp[ijkp+1]*(v[ijk+1] - v[ijk])*dpi;
                wdudp_up[ij] = 0.0;
                wdvdp_up[ij] = 0.0;
            } // end if
            else if (k==kmax-1){
                wdudp_dn[ij] = 0.0;
                wdvdp_dn[ij] = 0.0;
                wdudp_up[ij] = 0.5*wp[ijkp]*(u[ijk] - u[ijk-1])*dpi;
                wdvdp_up[ij] = 0.5*wp[ijkp]*(v[ijk] - v[ijk-1])*dpi;
            } // end else if
            else{
                wdudp_dn[ij] = 0.5*wp[ijkp+1]*(u[ijk+1] - u[ijk])*dpi;
                wdvdp_dn[ij] = 0.5*wp[ijkp+1]*(v[ijk+1] - v[ijk])*dpi;
                wdudp_up[ij] = 0.5*wp[ijkp]*(u[ijk] - u[ijk-1])*dpi;
                wdvdp_up[ij] = 0.5*wp[ijkp]*(v[ijk] - v[ijk-1])*dpi;
            } // end else
            e_dry[ij]  = (gz[ijkp+1]+gz[ijkp])/2.0 + ke[ijk];
            u_vort[ij] = u[ijk] * (vort[ijk]+f[ij]);
            v_vort[ij] = v[ijk] * (vort[ijk]+f[ij]);
        } // end j loop
    } // end i loop
    return;
}
