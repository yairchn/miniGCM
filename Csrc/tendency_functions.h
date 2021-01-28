#pragma once
// #include "grid.h"


void rhs_T(double cp,
           double* restrict p,
           double* restrict T,
           double* restrict gz,
           double* restrict wp,
           double* restrict T_mp,
           double* restrict T_sur,
           double* restrict T_forc,
           double* restrict rhs_T,
           int imax,
           int jmax,
           int kmax,
           Py_ssize_t k)
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
                wT_dn  = 0.5*wp[i,j,k+1]*(T[i,j,k]  +T[i,j,k])*dpi;
                wT_up  = 0.5*wp[i,j,k]  *(T[i,j,k]  +T[i,j,k-1])*dpi;
            } // end else if
            else{
                wT_dn  = 0.5*wp[i,j,k+1]*(T[i,j,k+1]  + T[i,j,k])*dpi;
                wT_up  = 0.5*wp[i,j,k]  *(T[i,j,k]    + T[i,j,k-1])*dpi;
            } // end else
        w_gz_dpi = wp[i,j,k+1]*(gz[i,j,k+1]-gz[i,j,k])*dpi/cp;
        rhs_T[i,j] = (wT_up - wT_dn - w_gz_dpi + T_mp[i,j,k] + T_forc[i,j,k] + T_sur[i,j]);
        } // End i loop
    } // end j loop
    return;
}


void rhs_qt(double* restrict p,
            double* restrict qt,
            double* restrict wp,
            double* restrict qt_mp,
            double* restrict qt_sur,
            double* restrict rhs_qt,
            int imax,
            int jmax,
            int kmax,
            Py_ssize_t k)
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
        } // End i loop
    } // end j loop
    return;
}
