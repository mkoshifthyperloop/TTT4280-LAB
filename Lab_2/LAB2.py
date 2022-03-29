import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from cmath import sin
from random import sample
from turtle import color
import math as m
import scipy.signal as sp

def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
        data = data[int(len(data)/3):]
    return sample_period, data

def find_time_delay(signal1, signal2, Fs):
    Fs = int(Fs)
    #lin = np.linspace(0, 1, Fs)
    #print(signal1.shape)
    cor = np.abs(sp.correlate(signal1, signal2, mode="full"))
    #print(cor.shape)
    corLin = np.linspace(-0.5,0.5,len(cor))
    #print(corLin)
    lags = sp.correlation_lags(signal1.size, signal2.size, mode="full")
    lag = lags[np.argmax(cor)]
    tau = lag/Fs
    return cor, corLin, lag

def find_Angle(n21, n31, n32):
    theta = np.arctan2(np.sqrt(3) * (n21+n31),(n21-n31-2*n32))
    
    return theta #if -n21+n31+2*n32 < 0 else theta+np.pi


# Import data from bin file
sample_period, data = raspi_import('160calib.bin')

#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

data = data/4096*3.3-3.3/2

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
print(num_of_samples)
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels
cor01, corlin01, s01= find_time_delay(data[:,1], data[:,0], 1/sample_period)
cor12, corlin12, s12= find_time_delay(data[:,2], data[:,1], 1/sample_period)
cor02, corlin02, s02= find_time_delay(data[:,2], data[:,0], 1/sample_period)
cor00, corlin00, as00 = find_time_delay(data[:,0], data[:,0], 1/sample_period)

time01 = s01*sample_period
time12 = s12*sample_period
time02 = s02*sample_period

print("samples: {} {} {} times: {} {} {} ".format(s01, s12, s02, time01, time12, time02))

theta = find_Angle(s01, s02, s12)
print(np.degrees(theta))

# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
plt.subplot(2, 2, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, data[:,0])
plt.plot(t, data[:,1])
plt.plot(t, data[:,2])

plt.subplot(2, 2, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum[:,0])), label="Mic 1") # get the power spectrum
plt.plot(freq, 20*np.log10(np.abs(spectrum[:,1])), label="Mic 2")
plt.plot(freq, 20*np.log10(np.abs(spectrum[:,2])), label="Mic 3")
plt.legend()

plt.subplot(2, 2, 3)
plt.title("Krysskorelasjon")
plt.plot(corlin01, cor01)
plt.plot(corlin12, cor12)
plt.plot(corlin02, cor02)

plt.subplot(2, 2, 4)
plt.title("Autokorrelasjonsfunksjonen")
plt.plot(corlin00, cor00)


plt.show()
