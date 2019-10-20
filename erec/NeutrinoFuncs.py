from numpy import pi, sqrt, exp, zeros, size, shape, array, arange
from numpy import trapz, interp, loadtxt, count_nonzero
from numpy.linalg import norm
import LabFuncs
from Params import *

#================================NeutrinoFuncs=================================#
# Contents:

#========================================Neutrino data=========================#
def GetNuFluxes(E_th,Atom):
    # Reads each neutrino flux data file
    # the energies are stored in E_nu_all, fluxes in Flux_all

    # Figure out which backgrounds give recoils above E_th
    E_r_max = MaxNuRecoilEnergies(Atom) # Max recoil energy for neutrino
    sel = range(1,n_nu_tot+1)*(E_r_max>E_th)
    sel = sel[sel!=0]-1
    n_nu = count_nonzero(E_r_max>E_th)
    E_nu_all = zeros(shape=(n_Enu_vals,n_nu))
    Flux_all = zeros(shape=(n_Enu_vals,n_nu))
    Flux_err = zeros(shape=(n_nu))
    Flux_norm = zeros(shape=(n_nu))
    Solar = zeros(n_nu,dtype=bool)

    ii = 0
    for s in sel:
        if mono[s]:
            E_nu_all[0,ii] = NuMaxEnergy[s]
            Flux_all[0,ii] = NuFlux[s]
        else:
            data = loadtxt(nufile_dir+nuname[s]+nufile_root,delimiter=',')
            E_nu_all[:,ii],Flux_all[:,ii] = data[:,0],data[:,1]
            Flux_all[:,ii] = Flux_all[:,ii]*NuFlux[s]

        Flux_norm[ii] = NuFlux[s]
        Flux_err[ii] = NuUnc[s] # Select rate normalisation uncertainties
        Solar[ii] = whichsolar[s]
        ii = ii+1
    NuBG = Neutrinos(n_nu,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err)
    return NuBG



# vacuum dominated MSW-Large mixing angle
# we can look at modifying this and allowing for energy dependence later
flav_Solar = array([0.55,0.45/2.0,0.45/2.0,0.0,0.0,0.0])

def dRdEe_nu(E_r,t,sol,E_nu,Flux,Atom,flav):
    N = Atom.NumberOfNeutrons
    Z = Atom.NumberOfProtons
    Q_W = 1.0*N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    m_N_keV = m_N_GeV*1.0e6 # nucleus mass in keV
    m_e_GeV = 5.109989461e-4

    ne = size(E_r)

    if sol:
        fMod = LabFuncs.EarthSunDistanceMod(t)
    else:
        fMod = 1.0

    gV = array([2*sinTheta_Wsq+0.5,2*sinTheta_Wsq-0.5,2*sinTheta_Wsq-0.5,
               2*sinTheta_Wsq+0.5,2*sinTheta_Wsq-0.5,2*sinTheta_Wsq-0.5])
    gA = array([0.5,-0.5,-0.5,-0.5,0.5,0.5])

    T = E_r/1000 # MeV
    Tmax = 2*E_nu**2.0/(2*E_nu+m_e_GeV*1000)

    dR = 0.0
    for ii in arange(0,6)[flav>0]:
        dRdE = zeros(shape=ne)
        As = (gV[ii]+gA[ii])**2.0
        Bs = (gV[ii]-gA[ii])**2.0
        Cs = (gA[ii]**2.0-gV[ii]**2.0)
        if Flux[1]>0.0:
            for i in range(0,ne):
                diff_sigma = (G_F_GeV**2.0*m_e_GeV/(2*pi))*(As+Bs*(1-T[i]/E_nu)**2.0+Cs*1000*m_e_GeV*T[i]/E_nu**2.0)
                diff_sigma *= (0.197e-13)**2.0*(1.0e-6)*1000.0/(1.0*N+1.0*Z)*(N_A)
                diff_sigma[T[i]>Tmax] = 0.0
                dRdE[i] = trapz(diff_sigma*Flux,x=E_nu)
        else:
            diff_sigma = (G_F_GeV**2.0*m_e_GeV/(2*pi))*(As+Bs*(1-T/E_nu[0])**2.0+Cs*1000*m_e_GeV*T/E_nu[0]**2.0)
            diff_sigma *= (0.197e-13)**2.0*(1.0e-6)*1000.0/(1.0*N+1.0*Z)*(N_A)
            diff_sigma[T>Tmax[0]] = 0.0
            dRdE = diff_sigma*Flux[0]*E_nu[0]*(diff_sigma>0.0) # for monochromatic nu's

        # Convert into /ton/year/keV
        dRdE = fMod*dRdE*(365.0*3600.0*24.0*1000.0)
        dRdE = dRdE*Z*flav[ii]

        dR += dRdE


    return dR




 #-----------------------------------------------------------------------------#
def MaxNuRecoilEnergies(Atom): # Max nuclear recoil energies
    m_N = 0.93141941*(Atom.MassNumber)*1.0e6
    E_r_max = 2*m_N*(1000.0*NuMaxEnergy)**2.0/(m_N+1000*NuMaxEnergy)**2.0
    return E_r_max


#===================================nu spectra=================================#
def dRdE_nu(E_r,t,sol,E_nu,Flux,Atom):
    N = Atom.NumberOfNeutrons
    Z = Atom.NumberOfProtons
    Q_W = 1.0*N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    m_N_keV = m_N_GeV*1.0e6 # nucleus mass in keV

    dRdE = zeros(shape=shape(E_r))
    FF = LabFuncs.FormFactorHelm(E_r,N+Z)**2.0
    ne = size(E_r)

    if Flux[1]>0.0:
        for i in range(0,ne):
            diff_sigma = (G_F_GeV**2.0 /(4.0*pi))*(Q_W**2.0)*m_N_GeV*(1.0 \
                        -(m_N_keV*E_r[i])/(2.0*(E_nu*1000.0)**2.0))*\
                        (0.197e-13)**2.0*(1.0e-6)*1000.0/(1.0*N+1.0*Z)*(N_A)
            diff_sigma[diff_sigma<0.0] = 0.0
            dRdE[i] = trapz(diff_sigma*Flux*FF[i],x=E_nu)
    else:
        for i in range(0,ne):
            diff_sigma = (G_F_GeV**2.0 /(4.0*pi))*(Q_W**2.0)*m_N_GeV*(1.0 \
                        -(m_N_keV*E_r[i])/(2.0*(E_nu[0]*1000.0)**2.0))*\
                        (0.197e-13)**2.0*(1.0e-6)*1000.0/(1.0*N+1.0*Z)*(N_A)
            if diff_sigma>0:
                dRdE[i] = diff_sigma*Flux[0]*E_nu[0] # for monochromatic nu's

    if sol:
        fMod = LabFuncs.EarthSunDistanceMod(t)
    else:
        fMod = 1.0

    # Convert into /ton/year/keV
    dRdE = fMod*dRdE*(365.0*3600.0*24.0*1000.0)
    return dRdE

def dRdEdO_isonu(E,E_nu,Flux,Atom):
    E_r = sqrt(E[:,0]**2 + E[:,1]**2 + E[:,2]**2) # Recoil energy
    dR = dRdE_nu(E_r,0.0,False,E_nu,Flux,Atom)/(4*pi)
    return dR

def dRdEdO_solarnu(E,t,E_nu,Flux,Atom,Loc): # Directional CEnuNS for Solar
    N = Atom.NumberOfNeutrons
    Z = Atom.NumberOfProtons
    Q_W = N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    m_N_keV = m_N_GeV*1.0e6 # nucleus mass in keV
    E_nu_keV = E_nu*1e3


    E_r = sqrt(E[:,0]**2 + E[:,1]**2 + E[:,2]**2) # Recoil energy
    x = zeros(shape=shape(E))
    x_sun = zeros(shape=shape(E))
    x[:,0] = E[:,0]/E_r # Recoil direction
    x[:,1] = E[:,1]/E_r
    x[:,2] = E[:,2]/E_r
    ne =size(E_r)
    dRdEdO = zeros(shape=ne)
    for i in range(0,ne):
        x_sun[i,:] = LabFuncs.SolarDirection(t[i],Loc)
    cos_th_sun = -(x_sun[:,0]*x[:,0]+x_sun[:,1]*x[:,1]+x_sun[:,2]*x[:,2])
    FF = LabFuncs.FormFactorHelm(E_r,N+Z)**2.0


    # CHROMATIC NEUTRINOS
    if Flux[1]>0.0:
        E_max = 2*m_N_keV*E_nu_keV[-1]**2.0/(m_N_keV+E_nu_keV[-1])**2
        i_range = range(0,ne)*(E_r<=E_max)
        i_sel = i_range[i_range!=0]
        for i in i_sel:
            costh = cos_th_sun[i]
            E_nu_min = sqrt(m_N_keV*E_r[i]/2.0)
            if costh>(E_nu_min/m_N_keV):
                Eps = 1.0/(costh/E_nu_min - 1.0/m_N_keV)
                diff_sigma = (G_F_GeV**2/(4*pi))*Q_W**2*m_N_GeV*\
                            (1-(m_N_keV*E_r[i])/(2*Eps**2))*(0.197e-13)**2.0\
                            *1e-6*1000/(N+1.0*Z)*(N_A)
                Eps = Eps*(Eps>E_nu_min)
                Eps = Eps*(Eps<E_nu_keV[-1])
                F_value = interp(Eps,E_nu_keV,Flux)
                dRdEdO[i] = diff_sigma*F_value*Eps**2.0/(1000*E_nu_min)*FF[i] # /kg/keV

    # MONOCHROMATIC NEUTRINOS
    else:
        E_max = 2*m_N_keV*E_nu_keV[0]**2.0/(m_N_keV+E_nu_keV[0])**2
        i_range = range(0,ne)*(E_r<=E_max)
        i_sel = i_range[i_range!=0]
        for i in i_sel:
            costh = cos_th_sun[i]
            E_nu_min = sqrt(m_N_keV*E_r[i]/2.0)
            costh_r = ((E_nu_keV[0]+m_N_keV)/E_nu_keV[0])*sqrt(E_r[i]/(2*m_N_keV))

            # just need to accept angles close enough to costh_r to be accurate
            # around 0.01 is enough to be disguised by most angular resolutions
            if abs((costh)-(costh_r))<0.01:
                Eps = E_nu_keV[0]
                diff_sigma = (G_F_GeV**2/(4*pi))*Q_W**2*m_N_GeV*\
                            (1-(m_N_keV*E_r[i])/(2*Eps**2))*(0.197e-13)**2.0\
                            *1e-6*1000/(N+1.0*Z)*(N_A)*FF[i]
                dRdEdO[i] = diff_sigma*(Flux[0]/1000.0)*E_nu_keV[0] # /kg/keV


    fMod = LabFuncs.EarthSunDistanceMod(t)
    dRdEdO = fMod*dRdEdO*3600*24*365*1000/(2*pi) # /ton/year
    return dRdEdO
