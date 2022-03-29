import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal


framerate = 40
max_pulse = 240
min_pulse = 40
hp_cutoff = min_pulse/60
lp_cutoff = max_pulse/60

def raspi_import(path, channels=5):
    data = np.loadtxt(path, delimiter=" ").T
    sample_period = 1/framerate
    return sample_period, data

def butter_coeff(cutoffFreq, sampleFreq, filterType='high', order=6):
    nyqFreq = 0.5 * sampleFreq
    normal_cutoff = cutoffFreq / nyqFreq
    return signal.butter(order, normal_cutoff, btype=filterType, analog=False)


def butter_filter(dataPoints, cutoffFreq, sampleFreq, filterType, order=6):
    b, a = butter_coeff(cutoffFreq, sampleFreq, filterType, order=order)
    return signal.filtfilt(b, a, dataPoints)

def calculate(filename):

    # Import data from bin file
    sample_period, data = raspi_import(filename)

    data = signal.detrend(data)

    data = butter_filter(data, hp_cutoff, framerate, "high")
    data = butter_filter(data, lp_cutoff, framerate, "low")

    # Generate time axis
    num_of_samples = data.shape[1]  # returns shape of matrix
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

    # Generate frequency axis and take FFT
    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.fft(data)
    
    pulse = []
    snr = []
    for spec in spectrum:
        pulse.append(abs(freq[np.abs(spec).argmax()])*60)
        snr.append(20*np.log(abs(np.abs(spec).max())/np.mean(np.abs(spec[np.where((freq >= 3) & (freq < 4))]))))
    return [t, data, freq, spectrum, pulse, snr]

def Plot(t, data, freq, spectrum):
    # Plot the results in two subplots
    # NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
    # If you want a single channel, use data[n-1] to get channel n
    plt.figure("Data")
    plt.subplot(2, 1, 1)
    plt.title("Time domain signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.plot(t, data[0], "r", label="R")
    plt.plot(t, data[1], "g", label="G")
    plt.plot(t, data[2], "b", label="B")
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    plt.title("Power spectrum of signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relative Power [dB]")
    plt.plot(freq[1:int(len(freq)/8)], 20*np.log(np.abs(spectrum[0].T))[1:int(len(freq)/8)], "r", label="R") # get the power spectrum
    plt.plot(freq[1:int(len(freq)/8)], 20*np.log(np.abs(spectrum[1].T))[1:int(len(freq)/8)], "g", label="G") # get the power spectrum
    plt.plot(freq[1:int(len(freq)/8)], 20*np.log(np.abs(spectrum[2].T))[1:int(len(freq)/8)], "b", label="B") # get the power spectrum
    plt.legend(loc="upper right")

    plt.show()
all_datas = []
pulse = []
for i in range(1,6):
    all_data = calculate(f"Magnus{i}.txt")
    print(f"Measurement nr {i}:")
    print(f"Pulse: {all_data[4]}")
    pulse.append(all_data[4])
    print(f"SNR: {all_data[5]}\n")
    all_datas.append(all_data)

pulse = np.array(pulse).T

for i, c in enumerate(["Red", "Green", "Blue"]):
    print(f"{c}: \n Mean: {np.mean(pulse[i])}\n std: {np.std(pulse[i])} BPM")
for all_data in all_datas:
    Plot(all_data[0], all_data[1], all_data[2], all_data[3])