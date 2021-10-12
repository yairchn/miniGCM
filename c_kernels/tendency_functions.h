// A collection of tendency functions computed in a given k (vertical) level
// as some are conditioned upon a planner spcteral conversion in that level

#pragma once
#include <math.h>

// T equation is in flux form
void rhs_T(double cp,
           double* restrict p,
           double* restrict gz,
           double* restrict T,
           double* restrict u,
           double* restrict v,
           double* restrict wp,
           double* restrict T_mp,
           double* restrict T_sur,
           double* restrict T_forc,
           double* restrict turbflux,
           double* restrict rhs_T,
           double* restrict u_T,
           double* restrict v_T,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax,
           ssize_t k)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    double dpi;
    double dwTdp_up;
    double dwTdp_dn;
    double T_turbfluxdiv;

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
            u_T[ij] = u[ijk] * T[ijk];
            v_T[ij] = v[ijk] * T[ijk];
            dwTdp_up = 0;
            dwTdp_dn = 0;
            T_turbfluxdiv = 0;
            dpi = 1.0/(p[ijkp+1] - p[ijkp]);
            rhs_T[ij] = 0.0;
            if (kmax-1>0){
                if (k==0){
                    dwTdp_dn = -0.5*wp[ijkp+1]*(T[ijk+1] + T[ijk])*dpi;
                    T_turbfluxdiv = -(turbflux[ijk+1] - turbflux[ijk])*dpi;
                } // end if
                else if (k==kmax-1){
                    dwTdp_up =  0.5*wp[ijkp]  *(T[ijk] + T[ijk-1])*dpi;
                    dwTdp_dn = -0.5*wp[ijkp+1]*(T[ijk] + T[ijk])*dpi;
                    T_turbfluxdiv = -(T_sur[ij] - turbflux[ijk])*dpi;
                } // end else if
                else { // if not @ boundaries you can access k+1 and k-1
                    dwTdp_up =  0.5*wp[ijkp]  *(T[ijk]   + T[ijk-1])*dpi;
                    dwTdp_dn = -0.5*wp[ijkp+1]*(T[ijk+1] + T[ijk])*dpi;
                    T_turbfluxdiv = -(turbflux[ijk+1] - turbflux[ijk])*dpi;
                } // end else
            } // end if
            else if (1==0){ // in a single layer we have a change in mass (ps)
                dwTdp_dn = -0.5*wp[ijkp+1]*(T[ijk] + T[ijk])*dpi;
                T_turbfluxdiv = -(T_sur[ij] - turbflux[ijk])*dpi;
            }
            rhs_T[ij] = T_turbfluxdiv + dwTdp_dn + dwTdp_up;
            //rhs_T[ij] -= 0.5*(wp[ijkp+1]+wp[ijkp])*(gz[ijkp+1]-gz[ijkp])*dpi/cp;
            //rhs_T[ij] -= wp[ijkp+1]*(gz[ijkp+1]-gz[ijkp])*dpi/cp;
            rhs_T[ij] += T_mp[ijk];
            rhs_T[ij] += T_forc[ijk];
        } // End j loop
    } // end i loop
    return;
}



// qt equation is in flux form
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
    double dwqtdp_dn;
    double dwqtdp_up;
    double qT_turbfluxdivdiv;

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
            dwqtdp_dn = 0;
            dwqtdp_up = 0;
            qT_turbfluxdivdiv = 0;
            dpi = 1.0/(p[ijkp+1] - p[ijkp]);

            if (kmax-1>0){
                if (k==0){
                    dwqtdp_dn = -0.5*wp[ijkp+1]*(qt[ijk+1] + qt[ijk])*dpi;
                    qT_turbfluxdivdiv = -(turbflux[ijk+1] - turbflux[ijk])*dpi;
                } // end if
                else if (k==kmax-1){
                    dwqtdp_up =  0.5*wp[ijkp]  *(qt[ijk] + qt[ijk-1])*dpi;
                    dwqtdp_dn = -0.5*wp[ijkp+1]*(qt[ijk] + qt[ijk])*dpi;
                    qT_turbfluxdivdiv = -(qt_sur[ij] - turbflux[ijk])*dpi;
                } // end else if
                else { // if not @ boundaries you can access k+1 and k-1
                    dwqtdp_up =  0.5*wp[ijkp]  *(qt[ijk]   + qt[ijk-1])*dpi;
                    dwqtdp_dn = -0.5*wp[ijkp+1]*(qt[ijk+1] + qt[ijk])*dpi;
                    qT_turbfluxdivdiv = -(turbflux[ijk+1] - turbflux[ijk])*dpi;
                } // end else
            } // end if
            else{ // in a single layer we have a change in mass (ps)
                dwqtdp_dn = -0.5*wp[ijkp+1]*(qt[ijk] + qt[ijk])*dpi;
                qT_turbfluxdivdiv = -(qt_sur[ij] - turbflux[ijk])*dpi;
            }
            rhs_qt[ij] = qT_turbfluxdivdiv + dwqtdp_dn + dwqtdp_up + qt_mp[ijk];
            u_qt[ij] = u[ijk] * qt[ijk];
            v_qt[ij] = v[ijk] * qt[ijk];
        } // End j loop
    } // end i loop
    return;
}

// u,v equations are in advective form
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
            wdudp_up[ij] = 0.0;
            wdvdp_up[ij] = 0.0;
            wdudp_dn[ij] = 0.0;
            wdvdp_dn[ij] = 0.0;
            if (kmax-1>0){
                if (k==0){
                    wdudp_dn[ij] = 0.5*wp[ijkp+1]*(u[ijk+1] - u[ijk])*dpi;
                    wdvdp_dn[ij] = 0.5*wp[ijkp+1]*(v[ijk+1] - v[ijk])*dpi;
                } // end if
                else if (k==kmax-1){
                    wdudp_up[ij] = 0.5*wp[ijkp]*(u[ijk] - u[ijk-1])*dpi;
                    wdvdp_up[ij] = 0.5*wp[ijkp]*(v[ijk] - v[ijk-1])*dpi;
                } // end else if
                else { // if not @ boundaries you can access k+1 and k-1
                    wdudp_up[ij] = 0.5*wp[ijkp]*(u[ijk] - u[ijk-1])*dpi;
                    wdvdp_up[ij] = 0.5*wp[ijkp]*(v[ijk] - v[ijk-1])*dpi;
                    wdudp_dn[ij] = 0.5*wp[ijkp+1]*(u[ijk+1] - u[ijk])*dpi;
                    wdvdp_dn[ij] = 0.5*wp[ijkp+1]*(v[ijk+1] - v[ijk])*dpi;
                } // end else
            } // end if
            e_dry[ij]  = (gz[ijkp+1]+gz[ijkp])/2.0 + ke[ijk];
            //e_dry[ij]  = gz[ijkp] + ke[ijk];
            u_vort[ij] = u[ijk] * (vort[ijk]+f[ij]);
            v_vort[ij] = v[ijk] * (vort[ijk]+f[ij]);
        } // end j loop
    } // end i loop
    return;
}
