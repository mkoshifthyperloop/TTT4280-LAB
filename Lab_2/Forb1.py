import numpy as np 
import scipy.signal as sp
import matplotlib.pyplot as plt

Fs = 1000
lin = np.linspace(0, 1, Fs)
x = np.sin(lin*np.pi)
y = np.sin((lin+3/Fs)*np.pi)

cor = np.abs(sp.correlate(x, y, mode="full"))
corLin = np.linspace(-0.5,0.5,len(cor))
lags = sp.correlation_lags(x.size, y.size, mode="full")
lag = lags[np.argmax(cor)]
tau = lag/Fs
print(tau)


fig, axs = plt.subplots(2)
axs[0].plot(lin, x)
axs[0].plot(lin, y)
axs[1].plot(corLin, cor)
plt.show()
