import cython
from concurrent.futures import ThreadPoolExecutor
from Grid cimport Grid
import numpy as np
cimport numpy as np
from Parameters cimport Parameters

# make sure that total moisture content is non-negative
cpdef set_min_vapour(qp,qbar):
    qtot = qp + qbar
    qtot[qtot<0] = 0
    return (qtot-qbar)

# Function for plotting KE spectra
cpdef keSpectra(Grid Gr, u, v):
    uk = Gr.grdtospec(u)
    vk = Gr.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())
    Ek = np.zeros(np.amax(Gr.l)+1)
    k = np.arange(np.amax(Gr.l)+1)
    for i in range(0,np.amax(Gr.l)):
        Ek[i] = np.sum(Esp[np.logical_and(Gr.l>=i-0.5 , Gr.l<i+0.5)])
    return [Ek,k]

cpdef inv_operator_vt(Grid Gr, Parameters Pr, miu, Omega, epsilon, zeta_bar_m, Der2_dir, cos_lat_sqr):
    # the function calculates a matrix which acts as the inverse of the operator
    # acting on the barocilinc component of the mean meridional velocity
    # the term in operator_vt that is related to the Hadley circulation (transportaion of momentum by the meridional circulation)
    hadley_term  = (1/Pr.Omega) * miu * zeta_bar_m
    Had_term     = np.diag(hadley_term)
    # Second derivative term
    Der2_term = (2/epsilon) * (np.diag(cos_lat_sqr)) * Der2_dir # assuming geostrophic balance
    # Coriolis term
    Cor_term   = 2*np.diag(miu.^2)
    # total
    operator_vt = Der2_term + Cor_term + Had_term
    inv_op_vt = np.inv(operator_vt)
    return inv_op_vt


# function [x,w] = lgwt(N,a,b)

# display('lgwt')

# % lgwt.m
# %
# % This script is for computing definite integrals using Legendre-Gauss
# % Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
# % [a,b] with truncation order N
# %
# % Suppose you have a continuous function f(x) which is defined on [a,b]
# % which you can evaluate at any x in [a,b]. Simply evaluate it at all of
# % the values contained in the x vector to obtain a vector f. Then compute
# % the definite integral using sum(f.*w);
# %
# % Written by Greg von Winckel - 02/25/2004
# N=N-1;
# N1=N+1; N2=N+2;

# xu=linspace(-1,1,N1)';

# % Initial guess
# y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);

# % Legendre-Gauss Vandermonde Matrix
# L=zeros(N1,N2);

# % Derivative of LGVM
# Lp=zeros(N1,N2);

# % Compute the zeros of the N+1 Legendre Polynomial
# % using the recursion relation and the Newton-Raphson method

# y0=2;

# % Iterate until new points are uniformly within epsilon of old points
# while max(abs(y-y0))>eps
    
    
#     L(:,1)=1;
#     Lp(:,1)=0;
    
#     L(:,2)=y;
#     Lp(:,2)=1;
    
#     for k=2:N1
#         L(:,k+1)=( (2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1) )/k;
#     end
    
#     Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);
    
#     y0=y;
#     y=y0-L(:,N2)./Lp;
    
# end

# % Linear map from[-1,1] to [a,b]
# x=(a*(1-y)+b*(1+y))/2;

# % Compute the weights
# w=(b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2;
# end