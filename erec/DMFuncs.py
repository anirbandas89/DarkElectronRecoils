#================================WIMPFuncs.py==================================#
# Created by Ciaran O'Hare 2019

# Description:

# Contents:

#==============================================================================#
import numpy as np
from numpy import pi, sqrt, exp, zeros, size, shape, trapz, arange, argmin, squeeze, ndim, reshape
from numpy.linalg import norm
from scipy.special import erf
import LabFuncs
from Params import *
import AtomicFuncs


#==============================================================================#

def LightMediator(q):
    F = (alph*m_e/q)**2.0
    return F**2.0

def HeavyMediator(q):
    F = 1.0
    return F

#---------------------------------- v_min -------------------------------------#
def vmin_ER(E_b,E_r,q,m_DM):
    # E_b = binding energy in keV
    # E_r = recoil energy in keV
    # q = momentum transfer in keV
    # m_chi = DM mass in MeV
    vmin = (E_r+E_b)/q + q/(2*m_DM*1000.0)
    return vmin*c_km # km/s

def vmin_NR(E_r,A,m_chi,delta=0):
    # E_r = recoil energy in keVr
    # A = nucleus mass number
    # m_chi = Wimp mass in GeV
    # delta = for inelastic scattering
    mu_p = 1.0e6*m_chi*m_p/(1.0e6*m_chi + m_p) # reduced proton mass
    m_N_keV = A*m_p # nucleus mass in keV
    mu_N_keV = 1.0e6*m_chi*m_N_keV/(1.0e6*m_chi + m_N_keV) # reduced nucleus mass
    v_min = sqrt(1.0/(2*m_N_keV*E_r))*(m_N_keV*E_r/mu_N_keV + delta)*c_km
    return v_min

#---------------------------------- E_max -------------------------------------#
def MaxWIMPEnergy(A,v_lab,m_chi,v_esc):
    # A = nucleus mass number
    # v_lab = Lab velocity in km/s
    # m_chi = Wimp mass in GeV
    # v_esc = Escape speed in km/s
    m_N = m_p*A
    mu_N = 1.0e6*m_N*m_chi/(1.0e6*m_chi+m_N)
    E_max_lim = 2.0*mu_N*mu_N*2.0*((v_esc+sqrt(sum(v_lab**2.0)))/c_km)**2.0/m_N
    return E_max_lim

#-------------------- Recoil  rate--------------------------------------------#
def NuclearRecoilRate_SI(E_r,HaloIntegral,A,sigma_p,m_chi,\
                            rho_0=SHMpp.LocalDensity):
    # E_r = Recoil energy in keVr
    # HaloIntegral = g(vmin) or fhat(vmin,q) for non-dir. or dir. experiment
    # A = Nucleus Mass Number
    # sigma_p = SI WIMP-proton cross section in cm^2
    # m_chi = WIMP mass in GeV
    # rho_0 = Local DM density

    mu_p = 1.0e6*m_chi*m_p/(1.0e6*m_chi + m_p) # reduced mass
    FF = LabFuncs.FormFactorHelm(E_r,A)**2.0 # Form Factor^2
    R0 = (c_cm*c_cm)*((rho_0*1.0e6*A*A*sigma_p)/(2*m_chi*GeV_2_kg*mu_p*mu_p))
    HaloIntegral = HaloIntegral/(1000.0*100.0) # convert to cm^-1 s

    # Compute rate = (Rate amplitude * HaloIntegral * form factor)
    dR = R0*HaloIntegral*FF
    dR = dR*seconds2year*1000.0 # convert to units of 1/(keVr ton year)
    return dR


def ElectronRecoilRate(Atom,E_r_vals,m_DM,sigma_e,DMFormFactor,\
                    vmin_fine,gmin_fine,rho_0=SHMpp.LocalDensity,nq=20):

    E_B_vals = Atom.BindingEnergies/1000.0
    nsh = size(E_B_vals)

    Efine,qfine,fion_fine = Atom.IonisationFormFactor()

    # Constants
    m_DM_keV = m_DM*1000 # keV
    ne = size(E_r_vals)
    m_N_kg = Atom.MassNumber*m_p_kg # keV
    N_T = 1.0/m_N_kg # kg
    mu = m_DM_keV*m_e/(m_e+m_DM_keV) # keV
    n_DM = rho_0*1e6/m_DM_keV # cm**-3.0

    vmax = ((850.0)*1000)/3.0e8 # natural units

    if ndim(gmin_fine)>1:
        nt = shape(gmin_fine)[0]
    else:
        nt = 1
        gmin_fine = reshape(gmin_fine,(1,size(gmin_fine)))

    dRdlnE = zeros(shape=(nt,ne))

    for it in range(0,nt):
        for sh in range(0,nsh): # over orbitals
            E_B = E_B_vals[sh]
            Emax = 0.5*m_DM_keV*vmax**2.0-E_B
            if Emax>E_r_vals[0]:
                imax = arange(0,ne)[E_r_vals<Emax][-1]
                for i in range(0,imax+1): # over energies
                    E_r = E_r_vals[i]
                    if E_r<(0.5*m_DM_keV*vmax**2.0-E_B):

                        # Form factor
                        qmax = m_DM_keV*vmax*(1.0+sqrt(1-2*(E_r+E_B)/(m_DM_keV*vmax**2.0)))
                        qmin = m_DM_keV*vmax*(1.0-sqrt(1-2*(E_r+E_B)/(m_DM_keV*vmax**2.0)))
                        qvals = logspace(log10(qmin),log10(qmax),nq)
                        fion = interp(qvals,qfine,fion_fine[argmin(abs(Efine-E_r)),:,sh])

                        # interpolate at required vmin and calculate cross section
                        vmin_vals = vmin_ER(E_B,E_r,qvals,m_DM)
                        gmin = interp(vmin_vals,vmin_fine,gmin_fine[it,:])/(1000*100)
                        qfunc = qvals*fion*gmin*DMFormFactor(qvals)
                        dsigma = (100*3.0e8)**2*(sigma_e/(8*mu**2.0))*trapz(qfunc,qvals) # cm^3

                        # rate
                        dRdlnE[it,i] += N_T*n_DM*\
                                AtomicFuncs.FermiFactor(E_r)*dsigma # kg^-1 s^-1
    dRdlnE *= seconds2year # kg^-1 yr^-1
    return squeeze(dRdlnE)
