import numpy as np

from astropy import constants as const
from astropy import units as u

from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.misc import derivative

import camb
from camb import model, initialpower

#### FROM SCHIVE PAPER ####
OMEGA_M = 0.284
OMEGA_B = 0.046 ### NOT SPECIFIED IN THE PAPER
OMEGA_C = OMEGA_M - OMEGA_B
OMEGA_L = 0.716
SIGMA_8 = 0.818
N_S = 0.962
LITTLE_H = 0.696
H0 = LITTLE_H*100*u.km/u.s/u.Mpc

RHO_CRIT = 3 * H0**2 / (8 * np.pi * const.G)
RHO_CRIT = RHO_CRIT.to(u.Msun/u.Mpc**3)
RHO_M = OMEGA_M * RHO_CRIT
DELTA_C = 1.68

def M_to_L(M):
    logL = 0.4*(51.60 - M)
    L = 10**(logL)
    return L

def L_to_M(L):
    return 51.6-2.5*np.log10(L)

# Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
ombh2 = OMEGA_B*LITTLE_H**2
omch2 = OMEGA_C*LITTLE_H**2
pars.set_cosmology(H0=H0.value, ombh2=ombh2, omch2=omch2, omk=0.0)
pars.InitPower.set_params(As=2.21826764445e-9, ns=N_S)
pars.set_for_lmax(2200, lens_potential_accuracy=1) # Default lmax in web tool

pars.set_matter_power(redshifts=[0], kmax=2000)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1000, npoints=1000)
s8 = np.array(results.get_sigma8())
print(s8, (SIGMA_8-s8)/(SIGMA_8))
print(z)

k = kh * LITTLE_H * 1/u.Mpc
P_k0 = pk[0] / LITTLE_H**3 * u.Mpc**3
P_k0 = interp1d(k, P_k0)

KMIN = k[0]
KMAX = k[-1]

def H(z, Omega_m=OMEGA_M, Omega_L=OMEGA_L, H0=H0):
    hubble = H0*np.sqrt(Omega_m*(1+z)**3 + Omega_L)
    return hubble

def D_g(z, Omega_m=OMEGA_M, Omega_L=OMEGA_L, H0=H0):
    prefactor = 5*Omega_m*H(z, Omega_m, Omega_L)/(2*H0)
    integrand = lambda z_prime: (1+z_prime)/(H(z_prime, Omega_m, Omega_L)/H0)**3
    integral, err = quad(integrand, z, np.inf) #Gives the same answer as integrating up to z=1000
    return prefactor * integral

def D_g_n(z):
    return D_g(z)/D_g(0)

def Delta_2(k, P_k):
    result = k**3 * P_k(k)*u.Mpc**3 / (2*np.pi**2)
    return result

def Theta_R(k, R):
    V_R = 4/3 * np.pi * R**3
    result = 4*np.pi/(V_R * k**3) * (-k*R*np.cos(k*R*u.rad) + np.sin(k*R*u.rad))
    return result

def Sigma_R(z, R):
    growth_factor_ratio = D_g(z)/D_g(0)
    P_kz = P_k0(k) * growth_factor_ratio**2
    P_kz = interp1d(k, P_kz)
    integrand = lambda kval: np.abs(Theta_R(kval/u.Mpc, R=R))**2 * 1/kval * Delta_2(kval/u.Mpc, P_kz)
    integral, err = quad(integrand, KMIN.value, KMAX.value, limit=100) #seems to improve error slightly
    sigmaR = np.sqrt(integral)
    return sigmaR

def Sigma_M(M):
    if type(M).__module__ != 'astropy.units.quantity':
        M = M * u.solMass
    R = np.cbrt(3*M/(4*np.pi*RHO_M))
    return Sigma_R(0, R)

Sigma_M = np.vectorize(Sigma_M)

def dSigmadM(sm, M): #TAKEN AT z=0
    log_sm = lambda logM: np.log(sm(np.exp(logM)))
    dlog = derivative(log_sm, np.log(M), dx=0.1)
    return sm(M) * dlog / M

def hmf_f(sigma, fn):
    delta_c = DELTA_C
    if fn == "PS":
        f = np.sqrt(2/np.pi) * (delta_c/sigma) * np.exp(-delta_c**2/(2*sigma**2))
    elif fn == "ST":
        A = 0.322
        a = 0.707
        p = 0.3
        f = A * np.sqrt(2*a/np.pi) * (1 + (sigma**2/(a*delta_c**2))**p) * (delta_c/sigma) * np.exp(-a*delta_c**2/(2*sigma**2))
    else:
        raise NotImplementedError
    return f

def dndM(M, fn, sigma, dsdM, z):
    sigma_z = sigma(M) * D_g_n(z)
    dsdM_z = dsdM(M) * D_g_n(z)
    prefactor = -(RHO_M/M) * (1/sigma_z) * dsdM_z
    return prefactor * hmf_f(sigma_z, fn)

def log_interp1d(xx, yy, sgn=1, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(sgn*yy)
    lin_interp = interp1d(logx, logy, kind=kind, fill_value='extrapolate')
    log_interp = lambda zz: sgn*np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def loglin_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    lin_interp = interp1d(logx, yy, kind=kind, fill_value='extrapolate')
    log_interp = lambda zz: lin_interp(np.log10(zz))
    return log_interp


Mvals2 = np.logspace(7, 15, 1000)
sm = Sigma_M(Mvals2)
sm = loglin_interp1d(Mvals2, sm)
dsdm = dSigmadM(sm, Mvals2)
dsdm = log_interp1d(Mvals2, dsdm, sgn=-1)

ZVALS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 13.75, 14, 15, 16, 17]
mass_fns = {}
for z in ZVALS:
    dndM_ST = dndM(Mvals2, "ST", sm, dsdm, z)
    mass_fns[z] = dndM_ST

np.savez('mass_fns_new.npz', Mvals=Mvals2, z5=mass_fns[5], z6=mass_fns[6], z7=mass_fns[7], z8=mass_fns[8], z9=mass_fns[9], z10=mass_fns[10],
         z11=mass_fns[11], z12=mass_fns[12], z1375=mass_fns[13.75], z13=mass_fns[13], z14=mass_fns[14], z15=mass_fns[15], z16=mass_fns[16], z17=mass_fns[17])
