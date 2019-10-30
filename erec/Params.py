#================================Params.py=====================================#
# Created by Ciaran O'Hare 2019

# Description:
# This file just sets up some of the parameters that are used throughout the
# project. and some classes that link things together.

#==============================================================================#

from __future__ import print_function
from numpy import array, sqrt, pi, exp, interp, loadtxt, zeros, shape, ones
from numpy import logspace, linspace, log10
from scipy.special import erf, erfi


# Constants
m_p = 0.9315*1e6
m_p_keV = 0.9315*1e6
m_e = 511.0 # keV
c_m = 2.99792458e8 # speed of light in m/s
c_cm = c_m*100.0 # speed of light in cm/s
c_km = c_m/1000.0 # speed of light in km/s
GeV_2_kg = 1.0e6*1.783e-33 # convert GeV to kg
alph = 1.0/137.0 # fine structure constant
m_p_kg = 1.660538782e-27 # amu in kg
a0 = 0.268173 # Bohr radius keV^-1
N_A = 6.02214e23 # Avocado's constant
sinTheta_Wsq = 0.2387e0 # sin^2(Theta_W) weinberg angle
G_F_GeV = 1.16637e-5 # GeV**-2 ! Fermi constan in GeV
Jan1 = 2458849.5 # January 1st 2020
seconds2year = 365.25*3600*24


#==============================================================================#
# Set Nucleus params
class Atom:
    def __init__(self,xi,N,Z,J,Sp,Sn,fion,E_B_vals,E_gap, eps, Vfactor):
        self.IsotopicFraction = xi
        self.NumberOfNeutrons = N
        self.NumberOfProtons = Z
        self.MassNumber = N+Z
        self.NuclearSpin = J
        self.ExpProtonSpin = Sp
        self.ExpNeutronSpin = Sp
        if J>0.0:
            self.SDEnhancement = (4.0/3.0)*((J+1.0)/J)*(Sp-Sn)**2.0
        self.IonisationFormFactor = fion
        self.BindingEnergies = E_B_vals
        self.BandGapEnergy = E_gap
        self.ElectronHoleMeanEnergy = eps
        self.VCellFactor = Vfactor
#==============================================================================#








#==============================================================================#
# Set parameters of halo models and streams
class Halo:
    def __init__(self,rho_0,v_LSR,sig_v,v_esc,v_pec,beta,eta):
        self.LocalDensity = rho_0
        self.RotationSpeed = v_LSR
        self.Dispersion = sig_v
        self.EscapeSpeed =  v_esc
        self.PeculiarVelocity = v_pec
        self.Normalisation = erf(v_esc/(sqrt(2.0)*sig_v))-\
                            sqrt(2.0/pi)*(v_esc/sig_v)*\
                            exp(-v_esc**2.0/(2.0*sig_v**2.0))

        self.SausageEta = eta
        if eta>0.0:
            self.SausageBeta = beta
            sigr=sqrt(3*v_LSR**2.0/(2.0*(3-2.0*beta)))
            sigphi=sqrt(3*v_LSR**2.0*(1-beta)/(2.0*(3-2.0*beta)))
            sigz=sqrt(3*v_LSR**2.0*(1-beta)/(2.0*(3-2.0*beta)))
            self.SausageDispersionTensor = array([sigr,sigphi,sigz])
            self.Normalisation = erf(v_esc/(sqrt(2.0)*sigr)) \
                    - sqrt((1.0-beta)/beta)\
                    *exp(-v_esc**2.0/(2.0*sigphi**2.0))\
                    *erfi(v_esc/(sqrt(2)*sigr)*sqrt(beta/(1-beta)))

# Standard Halo Model (old parameters)
SHM = Halo(0.3,
        220.0,
        156.0,
        544.0,
        array([11.1,12.2,7.3]),
        0.0,
        0.0)

# Standard Halo Model++
SHMpp = Halo(0.55,
        233.0,
        164.8,
        528.0,
        array([11.1,12.2,7.3]),
        0.9,
        0.2)

####

class Stream:
    def __init__(self,v1,v2,v3,sig1,sig2,sig3):
        self.Velocity = array([v1,v2,v2])
        self.Dispersion = array([sig1,sig2,sig3])

S1stream = Stream(-29.6,-297.4,-72.8,82.6, 26.9, 58.5)
S2stream = Stream(6.0, 166.7, -242.8,48.6, 13.5, 26.0)
#S2stream_b = Stream(-70.9, 153.3, 161.5, 83.9, 29.6, 71.5)
#==============================================================================#










#==============================================================================#
# Current number of neutrinos sources:
n_nu_tot = 15
# Neutrino files names:
nufile_root = ".txt"
nufile_dir = "../data/neutrinos/"
nuname = ["" for x in range(0,n_nu_tot)]
nuname[0] = "pp"
nuname[1] = "pep"
nuname[2] = "hep"
nuname[3] = "7Be1"
nuname[4] = "7Be2"
nuname[5] = "8B"
nuname[6] = "13N"
nuname[7] = "15O"
nuname[8] = "17F"
nuname[9] = "DSNB"
nuname[10] = "Atm"
nuname[11] = "GeoU"
nuname[12] = "GeoTh"
nuname[13] = "GeoK"
nuname[14] = "Reactor"
n_Enu_vals = 1000
# Mark which neutrinos are monochromatic
mono = zeros(n_nu_tot,dtype=bool)
mono[[1,3,4]] = True

# Set which neutrinos are Solar
whichsolar = zeros(n_nu_tot,dtype=bool)
whichsolar[0:8] = True

# Neutrino max energies (MeV):
NuMaxEnergy = array([0.42341,1.44,18.765,0.3843,0.8613,16.34,1.193,\
                    1.7285,1.7365,91.201,981.75
                    ,4.54,2.33,1.3572,\
                    1.1418e1])

# Neutrino fluxes (cm-2 s-1 MeV-1) and uncertainties (%):
# (from Vinyoles et al (2017) Barcelona GS98 SSM)
NuFlux = array([5.98e10,1.44e8,7.98e3,4.93e8,4.50e9,5.16e6,\
                        2.78e8,2.05e8,5.29e6,85.7,10.54,\
                        3808776.91874,3352686.94783,21639789.2056,\
                        208537.673299])
NuUnc = array([0.006, 0.01, 0.3,0.06, 0.06, 0.02, 0.15 ,\
                        0.17 ,0.2 ,0.5, 0.25,\
                        0.2,0.257,0.168,\
                        0.08])

# Collect neutrino parameters:
class Neutrinos:
    def __init__(self,n_nu,solar_label,energies,fluxes,\
                    normlisations,uncertainties):
        self.Flux = fluxes
        self.Energy = energies
        self.Uncertainties = uncertainties*normlisations
        self.Normalisations = normlisations
        self.NumberOfNeutrinos = n_nu
        self.SolarLabel = solar_label

    def RecoilDistribution(self,RD):
        self.RD = RD
#==============================================================================#



#==============================================================================#
# Location class only has latitude and longitude at the moment
class Location:
    def __init__(self,lat,lon):
        self.Latitude = lat
        self.Longitude = lon

Boulby = Location(54.5591,0.8310)
GranSasso = Location(42.4691, 13.5654)
Kamioka = Location(36.2381, 137.1863)
SNOlab = Location(46.4719, -81.1868)
Stawell = Location(-37.0576, 142.7754)
Oahu = Location(21.4389, -158.0001)
GuantanamoBay = Location(20.0117, -75.1216)
Pyongyang = Location(39.0392, 125.7625)
#------------------------------------------------------------------------------#
