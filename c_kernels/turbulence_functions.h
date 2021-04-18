// computing the pointwise (in physical grid) turbulent flux following
// "Tatcher and Jablonowski, 2016"

#pragma once
#include <math.h>

void vertical_turbulent_flux(
           double g,
           double Ch,
           double Cq,
           double kappa,
           double p_ref,
           double Ppbl,
           double Pstrato,
           double* restrict p,
           double* restrict gz,
           double* restrict T,
           double* restrict qt,
           double* restrict u,
           double* restrict v,
           double* restrict wTh,
           double* restrict wqt,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double windspeed_up;
    double windspeed_dn;
    double za;
    double Kh_dn;
    double Kh_up;
    double Kq_dn;
    double Kq_up;
    double dpi;
    double Th_up;
    double Th_dn;

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
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                const ssize_t ijkp = ishift_p + jshift_p + k;
                // eq. (17) Tatcher and Jablonowski 2016
                windspeed_up = sqrt(u[ijk]*u[ijk]     + v[ijk]*v[ijk]);
                windspeed_dn = sqrt(u[ijk+1]*u[ijk+1] + v[ijk+1]*v[ijk+1]);
                za = gz[ijkmax_p-1]/g/2.0;
                // Ke is on pressure levels
                Kq_up = Cq*windspeed_up*za*exp(-pow(fmax(Ppbl - p[ijkp],0.0)/Pstrato,2.0));
                Kq_dn = Cq*windspeed_dn*za*exp(-pow(fmax(Ppbl - p[ijkp+1],0.0)/Pstrato,2.0));
                Kh_up = Ch*windspeed_up*za*exp(-pow(fmax(Ppbl - p[ijkp],0.0)/Pstrato,2.0));
                Kh_dn = Ch*windspeed_dn*za*exp(-pow(fmax(Ppbl - p[ijkp+1],0.0)/Pstrato,2.0));
                if (k==0){
                    wTh[ijk]   =  0.0;
                    wqt[ijk]   =  0.0;
                } // end if
                else{
                    Th_dn = T[ijk]  *pow((p[ijkp]   + p[ijkp+1])/2.0/p_ref, kappa);
                    Th_up = T[ijk-1]*pow((p[ijkp-1] + p[ijkp])/2.0/p_ref, kappa);
                    dpi = 2.0/(p[ijkp+1]-p[ijkp-1]);
                    wTh[ijk] = -Kh_up*(Th_dn - Th_up)*dpi;
                    wqt[ijk] = -Kq_up*(qt[ijk] - qt[ijk-1])*dpi;
                } // end else
            } // end k loop
        } // end j loop
    } // end i loop

    return;
}