from cmath import sin
from random import sample
from turtle import color
from matplotlib.pyplot import plot, subplot
import numpy as np
import math as m
import matplotlib.pyplot as plt


f_s=4000
x= np.linspace(-f_s/2+1, f_s/2, f_s)

spacing= np.linspace(-1,1, len(x))

signal1=np.sinc(spacing)
signal2=np.sinc(spacing+100/f_s)

r_xx = np.correlate(signal1, signal2, mode="same")
#print(r_xx)
maxR=(np.argmax(np.abs(r_xx))-f_s/2)/f_s


print(maxR)
subplot(2,1,1)
plt.plot(x , signal1)
plt.plot(x, signal2)
subplot(2,1,2)
plt.plot(x,abs(r_xx))
plt.show()





    