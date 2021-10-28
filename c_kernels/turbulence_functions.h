// computing the pointwise (in physical grid) turbulent flux following
// "Tatcher and Jablonowski, 2016"

#pragma once
#include <math.h>

void vertical_turbulent_flux(
           double cp,
           double lv,
           double g,
           double Dh,
           double Dq,
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
           double* restrict wE,
           double* restrict wqt,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double wT;
    double wKe;
    double windspeed;
    double za;
    double Kh;
    double Kq;
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
            windspeed = sqrt(u[ijkmax]*u[ijkmax] + v[ijkmax]*v[ijkmax]);
            za = gz[ijkmax_p-1]/g/2.0;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                const ssize_t ijkp = ishift_p + jshift_p + k;
                if (k==0){
                    wE[ijk]   =  0.0;
                    wqt[ijk]   =  0.0;
                } // end if
                else{
                    // Ke is on pressure levels
                    // eq. (17) Tatcher and Jablonowski 2016
                    Kq = Dq*windspeed*za*exp(-pow( (fmax(Ppbl - p[ijkp],0.0)/Pstrato),   2.0));
                    Kh = Dh*windspeed*za*exp(-pow( (fmax(Ppbl - p[ijkp],0.0)/Pstrato),   2.0));
                    Th_dn = T[ijk]  *pow((p[ijkp]   + p[ijkp+1])/2.0/p_ref, kappa);
                    Th_up = T[ijk-1]*pow((p[ijkp-1] + p[ijkp])/2.0/p_ref, kappa);
                    dpi = 2.0/(p[ijkp+1]-p[ijkp-1]); // pressure differnece from mid-layer values for ijk
                    wqt[ijk] = -Kq*(qt[ijk] - qt[ijk-1])*dpi;
                    wT = -Kh*(Th_dn - Th_up)*dpi;
                    wKe = 0;
                    wE[ijk] = cp*wT + lv*wqt[ijk] + wKe;
                } // end else
            } // end k loop
        } // end j loop
    } // end i loop

    return;
}

// STRUCTURE
//                                    (w'T'_2          -        w'T'_1)/(p2-p1)
// dT/dt_1  = -d/dp(K*dT/dp) = -(2*K*(T_2-T_1)/(p3-p1)                     - 0)/(p2-p1)
//                                    (w'T'_3          -        w'T'_2)/(p3-p2)
// dT/dt_2  = -d/dp(K*dT/dp) = -(2*K*(T_3-T_2)/(ps-p2) - 2*K*(T2-T1)/(p3-p1))/(p3-p2)
//                                    (w'T'_3          -        w'T'_2)/(p3-p2)
// dT/dt_3  = -d/dp(K*dT/dp) = -(surf_flux             - 2*K*(T3-T2)/(ps-p2))/(ps-p3)
// ------------ p1 w'T'_1 = 0.0
//  - - - - - -                                  (w'T'_2-w'T'_1)/(p2-p1)
// ------------ p2 w'T'_2 = -K_e(T2- T1)/(p3-p1)
//  - - - - - -                                  (w'T'_3-w'T'_2)/(p3-p2)
// ------------ p3 w'T'_3 = -K_e(T3- T2)/(ps-p2)
//  - - - - - -                                  (w'T'_s-w'T'_3)/(ps-p3)
// ------------ ps w'T'_s = surf_flux
