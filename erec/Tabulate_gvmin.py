from numpy import *
from HaloFuncs import *
from Params import *

# v_mins
n = int(input("Number of velocities between 0.0 and 800.0 km/s : "))
v_min = linspace(0.01,800.0,n)

# Times
ndays = int(input("Number of times between Jan 1 and Dec 31 km/s : "))
days = linspace(0.0,365.0-365.0/ndays,ndays)

sig_iso = SHMpp.Dispersion*array([1,1,1])

# Calculate everything
gmin_Iso = zeros(shape=(ndays,n))
gmin_Iso_gf = zeros(shape=(ndays,n))
gmin_Saus = zeros(shape=(ndays,n))
gmin_Saus_gf = zeros(shape=(ndays,n))
gmin_S1 = zeros(shape=(ndays,n))
gmin_S1_gf = zeros(shape=(ndays,n))
gmin_S2 = zeros(shape=(ndays,n))
gmin_S2_gf = zeros(shape=(ndays,n))
for i in range(0,ndays):
    gmin_Iso[i,:] = gvmin_Isotropic(v_min,day=days[i])
    # gmin_Iso_gf[i,:] = gvmin_Triaxial(v_min,day=days[i],sig=sig_iso,GravFocus=True)
    #
    # gmin_Saus[i,:] = gvmin_Triaxial(v_min,day=days[i])
    # gmin_Saus_gf[i,:] = gvmin_Triaxial(v_min,day=days[i],GravFocus=True)
    #
    # gmin_S1[i,:] = gvmin_Triaxial(v_min,day=days[i],sig=S1stream.Dispersion,v_shift=S1stream.Velocity)
    # gmin_S1_gf[i,:] = gvmin_Triaxial(v_min,day=days[i],sig=S1stream.Dispersion,v_shift=S1stream.Velocity,GravFocus=True)
    #
    # gmin_S2[i,:] = gvmin_Triaxial(v_min,day=days[i],sig=S2stream.Dispersion,v_shift=S2stream.Velocity)
    # gmin_S2_gf[i,:] = gvmin_Triaxial(v_min,day=days[i],sig=S2stream.Dispersion,v_shift=S2stream.Velocity,GravFocus=True)

    print('day = ',i+1,'of',ndays,sum(gmin_S1[i,:]),sum(gmin_S1_gf[i,:]))


savetxt('../data/gvmin/gvmin_Halo.txt',vstack((v_min,gmin_Iso)),delimiter='\t',fmt="%1.12f")
# savetxt('../data/gvmin/gvmin_Halo_GF.txt',vstack((v_min,gmin_Iso_gf)),delimiter='\t',fmt="%1.12f")
# savetxt('../data/gvmin/gvmin_Saus.txt',vstack((v_min,gmin_Saus)),delimiter='\t',fmt="%1.12f")
# savetxt('../data/gvmin/gvmin_Saus_GF.txt',vstack((v_min,gmin_Saus_gf)),delimiter='\t',fmt="%1.12f")
# savetxt('../data/gvmin/gvmin_S1.txt',vstack((v_min,gmin_S1)),delimiter='\t',fmt="%1.12f")
# savetxt('../data/gvmin/gvmin_S1_GF.txt',vstack((v_min,gmin_S1_gf)),delimiter='\t',fmt="%1.12f")
# savetxt('../data/gvmin/gvmin_S2.txt',vstack((v_min,gmin_S2)),delimiter='\t',fmt="%1.12f")
# savetxt('../data/gvmin/gvmin_S2_GF.txt',vstack((v_min,gmin_S2_gf)),delimiter='\t',fmt="%1.12f")
