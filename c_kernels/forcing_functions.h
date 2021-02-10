// computing the pointwise (in physical grid) forcing following
// "Held Suarez, 1994"

#pragma once
#include <math.h>

void focring_hs(
           double kappa,
           double p_ref,
           double sigma_b,
           double k_a,
           double k_b,
           double k_f,
           double k_s,
           double Dtheta_z,
           double T_equator,
           double DT_y,
           double* restrict p,
           double* restrict T,
           double* restrict T_bar,
           double* restrict sin_lat,
           double* restrict cos_lat,
           double* restrict u,
           double* restrict v,
           double* restrict u_forc,
           double* restrict v_forc,
           double* restrict T_forc,
           ssize_t imax,
           ssize_t jmax,
           ssize_t kmax)
           {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    double p_half;
    double sigma_ratio;
    double k_T;
    double k_v;

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
                p_half = (p[ijkp]+p[ijkp+1])/2.0;
                T_bar[ijk] = fmax(((T_equator - DT_y*sin_lat[ij]*sin_lat[ij] -
                                Dtheta_z*log(p_half/p_ref)*cos_lat[ij]*cos_lat[ij])*
                                pow(p_half/p_ref, kappa)),200.0);


                sigma_ratio = fmax((p_half/p[ijkmax_p]-sigma_b)/(1-sigma_b),0.0);
                k_T = k_a + (k_s-k_a)*sigma_ratio*pow(cos_lat[ij],4.0);
                k_v = k_b +  k_f*sigma_ratio;

                u_forc[ijk] = -k_v *  u[ijk];
                v_forc[ijk] = -k_v *  v[ijk];
                T_forc[ijk] = -k_T * (T[ijk] - T_bar[ijk]);

            } // end k loop
        } // end j loop
    } // end i loop

    return;
}

void print_c(double* T,
             double* var,
             double* var_2d,
             ssize_t imax,
             ssize_t jmax,
             ssize_t kmax)
             {

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    // for(ssize_t i=imin;i<imax*jmax;i++){
    //     printf("%ld\n", i);
    //     printf("%f\n",var[i]);
    // } // end i loop

    // for(ssize_t i=imin;i<imax;i++){
    //         const ssize_t ishift = i*istride ;
    //         for(ssize_t j=jmin;j<jmax;j++){
    //             const ssize_t jshift = j*jstride;
    //             for(ssize_t k=kmin;k<kmax;k++){
    //                 const ssize_t ijk = ishift + jshift + k ;

    const ssize_t ij =  150 * jmax + 220;
    const ssize_t ijk = 150*jmax*kmax + 220*kmax + 2;
    // printf("%f\n",T[ijk]);
    // printf("%f\n",var_2d[ij]);
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*jmax*kmax;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t ij = i * jmax + j;
            const ssize_t jshift = j*kmax;
            const ssize_t ijkmax = ishift + jshift + kmax;
            printf("%ld\n", ijkmax);
            var_2d[ij] = T[ijkmax-1];
            // for(ssize_t k=kmin;k<kmax;k++){
            //     const ssize_t ijk = ishift + jshift + k;
            //     // const ssize_t ijk = i*jmax*kmax + j*kmax + k;
            //     // ijk = i*J*K + j*K + k
            //     // const ssize_t ijkmax = ishift + jshift + kmax;
            //     printf("%ld\n", i);
            //     printf("%ld\n", j);
            //     printf("%ld\n", k);
            //     printf("%ld\n", ijk);
            //     printf("%lf\n", T[ijk]);
            //     // var[ijk] = T[ijk];
            //     // printf("%ld\n", ijkmax);
            //     // printf("%f\n",var[ijkmax]);
            //     // var[ijk] = T[ijk];

            // } // end k loop
        } // end j loop
    } // end i loop

    // scanf("%lf\n",var);
    // T_forc[i,j,k] = T[i,j,k];
    // printf("%ld\n", i);
    // printf("%ld\n", j);
    // printf("%ld\n", k);
    // printf("%f\n",var[0]);
    // printf("%f\n",var[1]);
    // printf("%f\n",var[2]);
    // printf("%f\n",var[3]);
    // printf("%f\n",var[4]);
    // printf("%f\n",var[5]);
    // printf("%f\n",var[6]);
    // printf("%f\n",var[7]);
    // printf("%f\n",var[8]);
    // printf("%f\n",T_forc[i,j,k]);

    return;
}
