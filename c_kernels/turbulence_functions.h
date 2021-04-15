// computing the pointwise (in physical grid) turbulent flux following
// "Tatcher and Jablonowski, 2016"

#pragma once
#include <math.h>

void vertical_turbulent_flux(
           double g,
           double c_e,
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
    double Ke_dn;
    double Ke_up;
    double dpi;
    double Th_up;
    double Th_md;
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
                Ke_up = c_e*windspeed_up*za*exp(-pow(fmax(Ppbl - p[ijkp],0.0)/Pstrato,2.0));
                Ke_dn = c_e*windspeed_dn*za*exp(-pow(fmax(Ppbl - p[ijkp+1],0.0)/Pstrato,2.0));
                if (k==0){
                    dpi = 2.0/(p[ijkp+2]-p[ijkp]);
                    Th_dn = T[ijk+1]*pow((p[ijkp+1]+p[ijkp+2])/2.0/p_ref, kappa);
                    Th_md = T[ijk]  *pow((p[ijkp]+p[ijkp+1])/2.0/p_ref, kappa);
                    wTh[ijk]   =  0.0;
                    wqt[ijk]   =  0.0;
                    wTh[ijk+1] = -Ke_dn*(Th_dn     - Th_md)*dpi;
                    wqt[ijk+1] = -Ke_dn*(qt[ijk+1] - qt[ijk])*dpi;
                } // end if
                else if (k==kmax-1){
                    dpi = 2.0/(p[ijkp+1]-p[ijkp-1]);
                    Th_md = T[ijk]  *pow((p[ijkp]   + p[ijkp+1])/2.0/p_ref, kappa);
                    Th_up = T[ijk-1]*pow((p[ijkp-1] + p[ijkp])/2.0/p_ref, kappa);
                    wTh[ijk]   = -Ke_up*(Th_md   - Th_up)*dpi;
                    wqt[ijk]   = -Ke_up*(qt[ijk] - qt[ijk-1])*dpi;
                    wTh[ijk+1] =  0.0;
                    wqt[ijk+1] =  0.0;
                } // end else if
                else{
                    Th_dn = T[ijk+1]*pow((p[ijkp+1] + p[ijkp+2])/2.0/p_ref, kappa);
                    Th_md = T[ijk]  *pow((p[ijkp]   + p[ijkp+1])/2.0/p_ref, kappa);
                    Th_up = T[ijk-1]*pow((p[ijkp-1] + p[ijkp])/2.0/p_ref, kappa);

                    dpi = 2.0/(p[ijkp+1]-p[ijkp-1]);
                    wTh[ijk] = -Ke_up*(Th_md   - Th_up)*dpi;
                    wqt[ijk] = -Ke_up*(qt[ijk] - qt[ijk-1])*dpi;

                    dpi = 2.0/(p[ijkp+2]-p[ijkp]);
                    wTh[ijk+1] = -Ke_dn*(Th_dn     - Th_md)*dpi;
                    wqt[ijk+1] = -Ke_dn*(qt[ijk+1] - qt[ijk])*dpi;
                } // end else
            } // end k loop
        } // end j loop
    } // end i loop

    return;
}