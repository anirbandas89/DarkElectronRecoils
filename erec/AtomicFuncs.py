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

def fion_He():
    dat = loadtxt('../data/fion/fion_He.txt')
    Efine = dat[0,:]
    qfine = dat[1,:]
    n = size(qfine)

    fion_fine = zeros(shape=(n,n,1))
    fion_fine[:,:,0] = 10.0**dat[2:(n+2),:]
    return Efine,qfine,fion_fine

def fion_Ge():
    dat = loadtxt('../data/fion/fion_Ge.txt')
    Efine = dat[0,:]
    qfine = dat[1,:]
    n = size(qfine)

    fion_fine = zeros(shape=(n,n,1))
    fion_fine[:,:,0] = 10.0**dat[2:(n+2),:]
    return Efine,qfine,fion_fine

def fion_Si():
    dat = loadtxt('../data/fion/fion_Si.txt')
    Efine = dat[0,:]
    qfine = dat[1,:]
    n = size(qfine)

    fion_fine = zeros(shape=(n,n,1))
    fion_fine[:,:,0] = 10.0**dat[2:(n+2),:]
    return Efine,qfine,fion_fine

def fion_Xe():
    dat = loadtxt('../data/fion/fion_Xe.txt')
    Efine = dat[0,:]
    qfine = dat[1,:]
    n = size(qfine)

    fion_fine = zeros(shape=(n,n,3))
    fion_fine[:,:,0] = 10.0**dat[2:(n+2),:]
    fion_fine[:,:,1] = 10.0**dat[(n+2):(2*n+2),:]
    fion_fine[:,:,2] = 10.0**dat[2*n+2:3*n+2,:]
    return Efine,qfine,fion_fine
#==============================================================================#


#==============================================================================#
# Some targets:
#           (xi,      N,   Z,    J,     Sp,      Sn,   fion, E_B, E_gap, Ehole_mean, Vfactor)
He4 =   Atom(1.0,     2,   2,   0.01,  0.000,  0.000,  fion_He, array([24.982257]), 0.0, 0.0, 0.0)
Xe131 = Atom(0.212,   77,  54,  1.5,  -0.038,  0.242,  fion_Xe, array([12.4,25.7,75.6]), 0.0, 0.0, 0.0)
Xe129 = Atom(0.265,   75,  54,  0.5,   0.046,  0.293,  fion_Xe, array([12.4,25.7,75.6]), 0.0, 0.0, 0.0)
Ge =  Atom(1.0,    40.64,  32,  0.0,   0.00,   0.000,  fion_Ge, array([0.0]), 0.67, 2.9, 1.8)
Si =  Atom(1.0,  14.0855,  14,  0.0,   0.00,   0.000,  fion_Si, array([0.0]), 1.11, 3.6, 2.0)

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
