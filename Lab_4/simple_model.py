import numpy as np


muabo = np.genfromtxt("./muabo.txt", delimiter=",")
muabd = np.genfromtxt("./muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 520 # Replace with wavelength in nanometres
blue_wavelength = 460 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def transmitance_calc(bvf, d):
    def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
    def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

    oxy = 0.8 # Blood oxygenation

    # Absorption coefficient ($\mu_a$ in lab text)
    # Units: 1/m
    mua_other = 25 # Background absorption due to collagen, et cetera
    mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
                + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
    mua = mua_blood*bvf + mua_other

    # reduced scattering coefficient ($\mu_s^\prime$ in lab text)
    # the numerical constants are thanks to N. Bashkatov, E. A. Genina and
    # V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
    # tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
    # Units: 1/m
    musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

    # mua and musr are now available as shape (3,) arrays
    # Red, green and blue correspond to indexes 0, 1 and 2, respectively

    # TODO calculate penetration depth
    C = np.sqrt((3*(musr + mua)*mua))

    transmitance = np.exp(-C*d)
    #print(transmitance*100)
    return transmitance

contrast = (abs(transmitance_calc(1,300*10**(-6))-transmitance_calc(0.01,300*10**(-6))))/transmitance_calc(0.01, 300*10**(-6))
print(contrast)