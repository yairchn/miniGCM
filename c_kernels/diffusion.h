// Numerical diffusion methods
// these are coded once fr 3D variables and once for 2D variables (surface pressure)
#pragma once
#include <math.h>

// 3D forward euler - first setp for Adams Bashforth
void hyperdiffusion(
           double dt,
           double efold,
           int truncation_number,
           double dissipation_order,
           complex double* laplacian,
           complex double* variable,
           int* shtns_l,
           ssize_t imax,
           ssize_t kmax)
           {
    const ssize_t imin = 0;
    const ssize_t kmin = 0;
    double HyperDiffusionFactor;
    double diffusion_factor;

    for(ssize_t i=imin;i<imax;i++){
        diffusion_factor = 1.0/efold*cpow(laplacian[i]/laplacian[-1],(dissipation_order/2.0));
        HyperDiffusionFactor = cexp(-dt*diffusion_factor);
        if (shtns_l[i]>=truncation_number) {
            HyperDiffusionFactor = 0.0;
        }
        const ssize_t ishift = i*kmax;
        for(ssize_t k=kmin;k<kmax;k++){
            const ssize_t ik = ishift + k;
            variable[ik] = HyperDiffusionFactor * variable[ik];
        } // end k loop
    } // end i loop
    return;
}