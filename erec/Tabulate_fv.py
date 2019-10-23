from numpy import *
from HaloFuncs import *
from Params import *

# v_mins
n = int(input("Number of velocities between 0.0 and 750.0 km/s : "))
v = linspace(0.01,750.0,n)

# Times
ndays = int(input("Number of times between Jan 1 and Dec 31 km/s : "))
days = linspace(0.0,365.0-365.0/ndays,ndays)

sig_iso = SHMpp.Dispersion*array([1,1,1])

# Calculate everything
fv_Iso = zeros(shape=(ndays,n))
fv_Iso_gf = zeros(shape=(ndays,n))
fv_Saus = zeros(shape=(ndays,n))
fv_Saus_gf = zeros(shape=(ndays,n))
fv_S1 = zeros(shape=(ndays,n))
fv_S1_gf = zeros(shape=(ndays,n))
fv_S2 = zeros(shape=(ndays,n))
fv_S2_gf = zeros(shape=(ndays,n))
for i in range(0,ndays):
    fv_Iso[i,:] = SpeedDist_Isotropic(v,day=days[i])
    # fv_Iso_gf[i,:] = SpeedDist_Triaxial(v,day=days[i],sig=sig_iso,GravFocus=True)
    #
    # fv_Saus[i,:] = SpeedDist_Triaxial(v,day=days[i])
    # fv_Saus_gf[i,:] = SpeedDist_Triaxial(v,day=days[i],GravFocus=True)
    #
    # fv_S1[i,:] = SpeedDist_Triaxial(v,day=days[i],sig=S1stream.Dispersion,v_shift=S1stream.Velocity)
    # fv_S1_gf[i,:] = SpeedDist_Triaxial(v,day=days[i],sig=S1stream.Dispersion,v_shift=S1stream.Velocity,GravFocus=True)
    #
    # fv_S2[i,:] = SpeedDist_Triaxial(v,day=days[i],sig=S2stream.Dispersion,v_shift=S2stream.Velocity)
    # fv_S2_gf[i,:] = SpeedDist_Triaxial(v,day=days[i],sig=S2stream.Dispersion,v_shift=S2stream.Velocity,GravFocus=True)

    print('day = ',i+1,'of',ndays,sum(fv_S1[i,:]),sum(fv_S1_gf[i,:]))


savetxt('../data/gvmin/gvmin_Halo.txt',vstack((v,fv_Iso)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_Halo_GF.txt',vstack((v,fv_Iso_gf)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_Saus.txt',vstack((v,fv_Saus)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_Saus_GF.txt',vstack((v,fv_Saus_gf)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_S1.txt',vstack((v,fv_S1)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_S1_GF.txt',vstack((v,fv_S1_gf)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_S2.txt',vstack((v,fv_S2)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_S2_GF.txt',vstack((v,fv_S2_gf)),delimiter='\t',fmt="%1.12f")
