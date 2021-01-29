#pragma once


void rhs_T(double cpi,
           double* restrict p,
           double* restrict gz,
           double* restrict T,
           double* restrict u,
           double* restrict v,
           double* restrict wp,
           double* restrict T_mp,
           double* restrict T_sur,
           double* restrict T_forc,
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
    double wT_dn;
    double wT_up;
    double dpi;
    double w_gz_dpi;

    for(ssize_t i=imin;i<imax;i++){
        for(ssize_t j=jmin;j<jmax;j++){
            dpi = 1.0/(p[i,j,k+1] - p[i,j,k]);
        	if (k==0){
                wT_dn  = 0.5*wp[i,j,k+1]*(T[i,j,k+1] + T[i,j,k])*dpi;
                wT_up  = 0.0;
            } // end if
            else if (k==kmax){
                wT_dn  = 0.5*wp[i,j,k+1] * (T[i,j,k]  + T[i,j,k])*dpi;
                wT_up  = 0.5*wp[i,j,k] * (T[i,j,k]  + T[i,j,k-1])*dpi;
            } // end else if
            else{
                wT_dn  = 0.5*wp[i,j,k+1]*(T[i,j,k+1]  + T[i,j,k])*dpi;
                wT_up  = 0.5*wp[i,j,k]  *(T[i,j,k]    + T[i,j,k-1])*dpi;
            } // end else
        w_gz_dpi = wp[i,j,k+1]*(gz[i,j,k+1]-gz[i,j,k])*dpi*cpi;
        rhs_T[i,j] = (wT_up - wT_dn - w_gz_dpi + T_mp[i,j,k] + T_forc[i,j,k] + T_sur[i,j]);
        u_T[i,j] = u[i,j,k] * T[i,j,k];
        v_T[i,j] = v[i,j,k] * T[i,j,k];
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
    double wQT_dn;
    double wQT_up;
    double dpi;

    for(ssize_t i=imin;i<imax;i++){
        for(ssize_t j=jmin;j<jmax;j++){
            dpi = 1.0/(p[i,j,k+1] - p[i,j,k]);
        	if (k==0){
                wQT_dn = 0.5*wp[i,j,k+1]*(qt[i,j,k+1]+ qt[i,j,k])*dpi;
                wQT_up = 0.0;
            } // end if
            else if (k==kmax){
                wQT_dn = 0.5*wp[i,j,k+1]*(qt[i,j,k] +qt[i,j,k])*dpi;
                wQT_up = 0.5*wp[i,j,k]  *(qt[i,j,k] +qt[i,j,k-1])*dpi;
            } // end else if
            else{
                wQT_dn = 0.5*wp[i,j,k+1]*(qt[i,j,k+1] + qt[i,j,k])*dpi;
                wQT_up = 0.5*wp[i,j,k]  *(qt[i,j,k]   + qt[i,j,k-1])*dpi;
            } // end else
        rhs_qt[i,j] = (wQT_up - wQT_dn + qt_mp[i,j,k] + qt_sur[i,j]);
        u_qt[i,j] = u[i,j,k] * qt[i,j,k];
        v_qt[i,j] = v[i,j,k] * qt[i,j,k];
        } // End j loop
    } // end i loop
    return;
}


void vertical_uv_fluxes(double* restrict p,
                        double* restrict gz,
                        double* restrict vort,
                        double* restrict div,
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
        for(ssize_t j=jmin;j<jmax;j++){
            dpi = 1.0/(p[i,j,k+1] - p[i,j,k]);
            if (k==0){
                wdudp_up[i,j,k] = wp[i,j,k+1]*(u[i,j,k+1] - u[i,j,k])*dpi;
                wdvdp_up[i,j,k] = wp[i,j,k+1]*(v[i,j,k+1] - v[i,j,k])*dpi;
                wdudp_dn[i,j,k] = 0.0;
                wdvdp_dn[i,j,k] = 0.0;
            } // end if
            else if (k==kmax){
                wdudp_up[i,j,k] = 0.0;
                wdvdp_up[i,j,k] = 0.0;
                wdudp_dn[i,j,k] = wp[i,j,k]*(u[i,j,k] - u[i,j,k-1])*dpi;
                wdvdp_dn[i,j,k] = wp[i,j,k]*(v[i,j,k] - v[i,j,k-1])*dpi;
            } // end else if
            else{
                wdudp_up[i,j,k] = wp[i,j,k+1]*(u[i,j,k+1] - u[i,j,k])*dpi;
                wdvdp_up[i,j,k] = wp[i,j,k+1]*(v[i,j,k+1] - v[i,j,k])*dpi;
                wdudp_dn[i,j,k] = wp[i,j,k]*(u[i,j,k] - u[i,j,k-1])*dpi;
                wdvdp_dn[i,j,k] = wp[i,j,k]*(v[i,j,k] - v[i,j,k-1])*dpi;
            } // end else
            e_dry[i,j]  = gz[i,j,k] + ke[i,j,k];
            u_vort[i,j] = u[i,j,k] * (vort[i,j,k]+f[i,j]);
            v_vort[i,j] = v[i,j,k] * (vort[i,j,k]+f[i,j]);
        } // end j loop
    } // end i loop
    return;
}
