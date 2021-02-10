// time stepping methods
// these are coded once fr 3D variables and once for 2D variables (surface pressure)
#pragma once
#include <math.h>

// 3D forward euler - first setp for Adams Bashforth
void forward_euler(
           double dt,
           double complex* variable_old,
           double complex* tendency,
           double complex* variable_new,
           ssize_t imax,
           ssize_t kmax)
           {
    const ssize_t imin = 0;
    const ssize_t kmin = 0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*kmax;
        for(ssize_t k=kmin;k<kmax;k++){
            const ssize_t ik = ishift + k;
            variable_new[ik] = variable_old[ik] + dt*tendency[ik];
        } // end j loop
    } // end i loop
    return;
}

// 3D second order  Adams Bashforth - second setp for Adams Bashforth
void adams_bashforth_2nd_order(
           double dt,
           double* restrict variable_old,
           double* restrict tendency_now,
           double* restrict tendency,
           double* restrict variable_new,
           ssize_t imax,
           ssize_t kmax)
           {
    const ssize_t imin = 0;
    const ssize_t kmin = 0;
    const double ab1 = 1.5;
    const double ab2 = -0.5;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*kmax;
        for(ssize_t k=kmin;k<kmax;k++){
            const ssize_t ik = ishift + k;
            variable_new[ik] = variable_old[ik] + dt*(ab1*tendency[ik] + ab2*tendency_now[ik]);
        } // end j loop
    } // end i loop
    return;
}

// 3D third order  Adams Bashforth
void adams_bashforth_3rd_order(
           double dt,
           double* restrict variable_old,
           double* restrict tendency_old,
           double* restrict tendency_now,
           double* restrict tendency,
           double* restrict variable_new,
           ssize_t imax,
           ssize_t kmax)
           {
    const ssize_t imin = 0;
    const ssize_t kmin = 0;
    const double ab1 =  23.0/12.0;
    const double ab2 = -16.0/12.0;
    const double ab3 =  5.0/12.0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*kmax;
        for(ssize_t k=kmin;k<kmax;k++){
            const ssize_t ik = ishift + k;
            variable_new[ik] = variable_old[ik] +
                dt*(ab1*tendency[ik] + ab2*tendency_now[ik] + ab3*tendency_old[ik]);
        } // end j loop
    } // end i loop
    return;
}

// 2D forward euler - first setp for Adams Bashforth
void forward_euler_2d(
           double dt,
           double* restrict variable_old,
           double* restrict tendency,
           double* restrict variable_new,
           ssize_t imax,
           ssize_t kmax)
           {
    const ssize_t imin = 0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ik = i*kmax + kmax;
        variable_new[ik] = variable_old[i] + dt*tendency[ik];
    } // end i loop
    return;
}

// 2D second order  Adams Bashforth - second setp for Adams Bashforth
void adams_bashforth_2nd_order_2d(
           double dt,
           double* restrict variable_old,
           double* restrict tendency_now,
           double* restrict tendency,
           double* restrict variable_new,
           ssize_t imax,
           ssize_t kmax)
           {
    const ssize_t imin = 0;
    const ssize_t kmin = 0;
    const double ab1 = 1.5;
    const double ab2 = -0.5;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ik = i*kmax + kmax;
        variable_new[ik] = variable_old[ik] + dt*(ab1*tendency[ik] + ab2*tendency_now[ik]);
    } // end i loop
    return;
}

// 2D third order Adams Bashforth
void adams_bashforth_3rd_order_2d(
           double dt,
           double* restrict variable_old,
           double* restrict tendency_old,
           double* restrict tendency_now,
           double* restrict tendency,
           double* restrict variable_new,
           ssize_t imax,
           ssize_t kmax)
           {
    const ssize_t imin = 0;
    const ssize_t kmin = 0;
    const double ab1 =  23.0/12.0;
    const double ab2 = -16.0/12.0;
    const double ab3 =  5.0/12.0;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ik = i*kmax + kmax;
        variable_new[ik] = variable_old[ik] +
            dt*(ab1*tendency[ik] + ab2*tendency_now[ik] + ab3*tendency_old[ik]);
    } // end i loop
    return;
}



