import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal


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

        data = data[int(len(data) / 4):int(len(data) - len(data)/4)]

    return sample_period, data


def snr(a):
    max = 20*np.log10(np.abs(np.max(a)))
    mean = 20*np.log10(np.abs(np.mean(a)))
    return max - mean


def calculate(filename):

    # Import data from bin file
    sample_period, data = raspi_import(filename)

    data = signal.detrend(data, axis=0)  # removes DC component for each channel
    sample_period *= 1e-6  # change unit to micro seconds

    data = data/4096*3.3

    # Generate time axis
    num_of_samples = data.shape[0]  # returns shape of matrix
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

    # Generate frequency axis and take FFT
    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)

    q_chan = data[:,3]
    i_chan = data[:,4]
    retData = [q_chan, i_chan]

    #q_chan = q_chan * signal.get_window("hamming", q_chan.shape[0])
    #i_chan = i_chan * signal.get_window("hamming", i_chan.shape[0])

    doppler_freq = np.fft.fft(i_chan + 1j*q_chan)
    doppler_shift = freq[np.abs(doppler_freq).argmax()]
    print(f"Doopler shift: {doppler_shift}, velocity: {doppler_shift/160}\n")
    print(f"SNR: {snr(doppler_freq)} ")

    return doppler_shift/160, doppler_freq, retData, t, freq

# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n

files = [0,0,0]
shifts = [0,0,0]
results = [[], [], []]
datas = [[], [], []]
ts = [0,0,0]
freqs = [0,0,0]
for i in range(0, 3):
    files[i]=f"135b{i+1}.bin"
    shifts[i], results[i], datas[i], ts[i], freqs[i] = calculate(files[i])

var = np.var(shifts)
std = np.std(shifts)

print(f"Variance: {var}, std: {std}")

plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(ts[0], datas[0][0])
plt.plot(ts[0], datas[0][1])
plt.legend(["I Channel", "Q channel"])

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
for i in range(0,3):
    plt.plot(freqs[0], 20*np.log10(np.abs(results[i]))) # get the power spectrum

plt.show()
