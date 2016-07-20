import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from astropy.stats import sigma_clip
from astropy.stats import LombScargle

####We start by opening the files and extracting the data#####
parameters = np.loadtxt("../../Jason_code_outputs/K02052.01/newfit.dat", usecols = ([1]))
transit_center = parameters[8]
period = parameters[9]
orbital_frequency = 1./period
time, flux, flux_err = np.loadtxt("../../Jason_code_outputs/K02052.01/K02052.01_res.dat", usecols = (0,1,2), unpack = True)

print transit_center, period, len(time)
frequency, power = LombScargle(time, flux, flux_err).autopower(minimum_frequency=10.**(-0.05)*orbital_frequency, maximum_frequency=10.**(0.8)*orbital_frequency)

m1_left = (np.abs(frequency - 0.98*orbital_frequency)).argmin()
m1_right = (np.abs(frequency - 1.02*orbital_frequency)).argmin()
m2_left = (np.abs(frequency - 1.98*orbital_frequency)).argmin()
m2_right = (np.abs(frequency - 2.02*orbital_frequency)).argmin()
m3_left = (np.abs(frequency - 2.98*orbital_frequency)).argmin()
m3_right = (np.abs(frequency - 3.02*orbital_frequency)).argmin()

print m1_left, m1_right, m2_left, m2_right, m3_left, m3_right

m1_freq = frequency[np.argmax(power[m1_left:m1_right]) + m1_left]
m2_freq = frequency[np.argmax(power[m2_left:m2_right]) + m2_left]
m3_freq = frequency[np.argmax(power[m3_left:m3_right]) + m3_left]
print m1_freq/orbital_frequency, m2_freq/orbital_frequency , m3_freq/orbital_frequency
m1 = LombScargle(time, flux, flux_err).power(m1_freq)
m2 = LombScargle(time, flux, flux_err).power(m2_freq)
m3 = LombScargle(time, flux, flux_err).power(m3_freq)

print m1,m2,m3

time2 = np.ma.mod(time, period)

data = np.ma.column_stack((time2,flux,flux_err))                     
data = data[np.lexsort((data[:,2],data[:,1],data[:,0]))]	

bins = np.linspace(0., period, 200)
digitized = np.digitize(data[:,0], bins)
bin_means = [data[:,1][digitized == i].mean() for i in range(len(bins))]
print len(bin_means), len(bins)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,period])
ax1.grid()
#ax1.set_ylim([0.97,1.03])
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data[:,0], data[:,1], 'o', ms=1)
ax1.plot(bins, bin_means, 'o', ms=5, color= 'red')

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency])
ax2.set_ylim([6.6e-5,0.7])
ax2.plot(frequency/orbital_frequency, power)
ax2.set_yscale('log')
#ax2.set_xticks([1,2,3,4,5,6])
#ax2.set_xticklabels([1,2,3,4,5,6])



plt.show()
