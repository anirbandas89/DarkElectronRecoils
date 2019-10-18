#================================WIMPFuncs.py==================================#
# Created by Ciaran O'Hare 2019

# Description:

# Contents:

#==============================================================================#

import numpy as np
from numpy import pi, sqrt, exp, zeros, size, shape, array, trapz, log10, abs
from numpy.linalg import norm
from scipy.special import erf, hyp2f1, gamma, factorial
import LabFuncs
from Params import *

#==============================================================================#


#==============================================================================#

# RHF Wave functions from Bunge et al. Atom. Data Nucl. Data Tabl. 53, 113 (1993).

# Radial part of physical space description
def R_nl(r,c_nlk,n_lk,Z_lk):
    x = r/a0
    nf = sqrt(factorial(2*n_lk)*1.0)
    R = 0.0
    for i in range(0,size(c_nlk)):
        R += c_nlk[i]*(2*Z_lk[i])**(n_lk[i]+0.5)/(a0**1.5*nf[i])*\
            (x**(n_lk[i]-1.0))*exp(-Z_lk[i]*x)
    return R

# Radial part of momentum space description
def chi_nl_sq(p,l,c_nlk,n_lk,Z_lk):
    nf = sqrt(factorial(2*n_lk)*1.0)
    chi = 0.0
    for i in range(0,size(c_nlk)):
        c = c_nlk[i]
        n = n_lk[i]
        Z = Z_lk[i]
        x = -a0**2.0*p**2.0/Z**2.0
        a1 = 0.5*(l+n+2)
        a2 = 0.5*(l+n+3)
        a3 = l+1.5
        chi += (pi**1.5)*c*(a0**1.5)*((a0*p)**l)*(2.0**(n-l+1.5))*\
            (Z**(-l-1.5))/nf[i]*gamma(l+n+2)*hyp2f1(a1,a2,a3,x)/gamma(a3)
    return chi**2.0


#==============================================================================#
# Ionisation form factors
# Currently only has Helium and Xenon

def f_nl_ion_sq(q,E_r,l,c_nlk,n_lk,Z_lk,np=20):
    ppr = sqrt(2*m_e*E_r)
    C = (2*l+1)*(ppr**2.0)/((4*pi**3.0)*q)
    #pvals = logspace(log10(abs(ppr-q[0])),log10(ppr+q[-1]),nfine)
    #chi = chi_nl_sq(pvals,l,c_nlk,n_lk,Z_lk)
    f = zeros(shape=size(q))
    for i in range(0,size(q)):
        pmin = abs(ppr-q[i])
        pmax = ppr+q[i]
        pvals = logspace(log10(pmin),log10(pmax),np)
        chi2 = chi_nl_sq(pvals,l,c_nlk,n_lk,Z_lk)
        f[i] = C[i]*trapz(pvals*chi2,pvals)

        #mask = (pvals<pmax)&(pvals>pmin)
        #f[i] = C[i]*trapz(chi[mask],pvals[mask])
    return f

def fion_He(qvals,E_r,sh,np=10):
    # s orbitals
    n_s = array([1]+[3]+[2]*2)
    Z_s = array([1.4595,5.3244,2.6298,1.7504])
    c_1s = array([1.1347900,-0.001613,-0.100506,-0.270779])

    orbitals = {
        0: f_nl_ion_sq(qvals,E_r,0,c_1s,n_s,Z_s,np=np),
    }
    return orbitals.get(sh)

def fion_Xe(qvals,E_r,sh,np=20):
    # s orbitals
    n_s = array([1]+[2]*2+[3]*3+[4]*3+[5]*4)
    Z_s = array([54.9179,47.2500,26.0942,68.1771,16.8296,12.0759,31.9030,8.0145,5.8396,14.7123,3.8555,2.6343,1.8124])
    c_1s = array([-0.965401,-0.040350,0.001890,-0.003868,-0.000263,0.000547,-0.000791,0.000014,-0.000013,-0.000286,0.000005,-0.000003,0.000001])
    c_2s = array([0.313912,0.236118,-0.985333,0.000229,-0.346825,0.345786,-0.120941,-0.005057,0.001528,-0.151508,-0.000281,0.000134,-0.000040])
    c_3s = array([-0.140382,-0.125401,0.528161,-0.000435,0.494492,-1.855445,0.128637,-0.017980,0.000792,0.333907,-0.000228,0.000191,-0.000037])
    c_4s = array([0.064020,0.059550,-0.251138,0.000152,-0.252274,1.063559,-0.071737,-0.563072,-0.697466,-0.058009,-0.018353,0.002292,-0.000834])
    c_5s = array([-0.022510,-0.021077,0.088978,-0.000081,0.095199,-0.398492,0.025623,0.274471,0.291110,0.011171,-0.463123,-0.545266,-0.167779])

    # p orbitals
    n_p = array([2]*2+[3]*3+[4]*3+[5]*4)
    Z_p = array([58.7712,22.6065,48.9702,13.4997,9.8328,40.2591,7.1841,5.1284,21.5330,3.4469,2.2384,1.14588])
    c_2p = array([0.051242,0.781070,0.114910,-0.000731,0.000458,0.083993,-0.000265,0.000034,0.009061,-0.000014,0.000006,-0.000002])
    c_3p = array([0.000264,0.622357,-0.009861,-0.952677,-0.337900,-0.026340,-0.000384,-0.001665,0.087491,0.000240,-0.000083,0.000026])
    c_4p = array([0.013769,-0.426955,0.045088,0.748434,0.132850,0.059406,-0.679569,-0.503653,-0.149635,-0.014193,0.000528,-0.000221])
    c_5p = array([-0.005879,0.149040,-0.018716,-0.266839,-0.031096,-0.024100,0.267374,0.161460,0.059721,-0.428353,-0.542284,-0.201667])

    # d orbitals
    n_d = array([3]*3+[4]*5)
    Z_d = array([19.9787,12.2129,8.6994,27.7398,15.9410,6.0580,4.0990,2.5857])
    c_4d = array([-0.013758,-0.804573,0.260624,0.00749,0.244109,0.597018,0.395554,0.039786])
    c_3d = array([0.220185,0.603140,0.194682,-0.014369,0.049865,-0.000300,0.000418,-0.000133])

    # Currently only doing 3 outermost shells, can go further just by adding
    # the next orbital to the list (but make sure to add another energy to the
    # Xenon131.BindingEnergies in Params.py or it won't work)
    orbitals = {
        0: f_nl_ion_sq(qvals,E_r,0,c_5s,n_s,Z_s,np=np),
        1: f_nl_ion_sq(qvals,E_r,1,c_5p,n_p,Z_p,np=np),
        2: f_nl_ion_sq(qvals,E_r,2,c_4d,n_d,Z_d,np=np),
    }
    return orbitals.get(sh)
#==============================================================================#


#==============================================================================#
# Some targets:
#           (xi,      N,   Z,    J,     Sp,      Sn,   fion, E_B)
He4 =   Atom(1.0,     2,   2, 0.01,  0.000,   0.000,fion_He, array([24.982257]))
Xe131 = Atom(0.212,  77,  54,  1.5, -0.038,   0.242,fion_Xe, array([12.4,25.7,75.6]))
Xe129 = Atom(0.265,  75,  54,  0.5,  0.046,   0.293,fion_Xe, array([12.4,25.7,75.6]))
# F19 =   Atom(1.0,    10,   9,  0.5,  0.421,   0.045,fion_F,array([]))
#==============================================================================#



#==============================================================================#
# Fermi factor for correcting outgoing plane wave approximation
def FermiFactor(E_r,Z_eff=1.0):
    ppr = sqrt(2*m_e*E_r)
    nu = Z_eff*(alph*m_e/ppr)
    F = 2*pi*nu/(1-exp(-2*pi*nu))
    return F
#==============================================================================#
