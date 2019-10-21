from numpy import *
from HaloFuncs import *
from Params import *
from AtomicFuncs import *

# s orbitals
n_s = array([1]+[3]+[2]*2)
Z_s = array([1.4595,5.3244,2.6298,1.7504])
c_1s = array([1.1347900,-0.001613,-0.100506,-0.270779])
n = int(input("Number of values"))

E_r_vals = logspace(-1.0,4.0,n)/1000.0 # keV
q_vals = logspace(-1.0,4.0,n)

np = 20

F1 = zeros(shape=(n,n))
for i in range(0,n):
    F1[i,:] = f_nl_ion_sq(q_vals,E_r_vals[i],0,c_1s,n_s,Z_s,np=np)
    print(i,'of',n)

savetxt('../data/fion/fion_He.txt',vstack((E_r_vals,q_vals,log10(F1))),delimiter='\t',fmt="%1.16f")
