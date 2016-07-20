import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from astropy.stats import sigma_clip
from astropy.stats import LombScargle


####We start by opening the files and extracting the data#####
parameters = np.loadtxt("../../Jason_code_outputs/K01546.01/n0.dat", usecols = ([1]))
transit_center = parameters[8]
period = parameters[9]
orbital_frequency = 1./period
time, flux, flux_err = np.loadtxt("../../Jason_code_outputs/K01546.01/newfile.dat", usecols = (0,1,2), unpack = True)

print transit_center, period
frequency, power = LombScargle(time, flux, flux_err).autopower(minimum_frequency=10.**(-0.05)*orbital_frequency, maximum_frequency=60.*orbital_frequency)

star1left = (np.abs(frequency - 43.5*orbital_frequency)).argmin()
star1right = (np.abs(frequency - 44.5*orbital_frequency)).argmin()

star2left = (np.abs(frequency - 45.5*orbital_frequency)).argmin()
star2right = (np.abs(frequency - 46.5*orbital_frequency)).argmin()

m1_left = (np.abs(frequency - 0.99*orbital_frequency)).argmin()
m1_right = (np.abs(frequency - 1.01*orbital_frequency)).argmin()
m2_left = (np.abs(frequency - 1.99*orbital_frequency)).argmin()
m2_right = (np.abs(frequency - 2.01*orbital_frequency)).argmin()
m3_left = (np.abs(frequency - 2.99*orbital_frequency)).argmin()
m3_right = (np.abs(frequency - 3.01*orbital_frequency)).argmin()

print m1_left, m1_right, m2_left, m2_right, m3_left, m3_right, star1left, star1right, star2left, 

s1_freq = frequency[np.argmax(power[star1left:star1right]) + star1left]
s2_freq = frequency[np.argmax(power[star2left:star2right]) + star2left]

m1_freq = frequency[np.argmax(power[m1_left:m1_right]) + m1_left]
m2_freq = frequency[np.argmax(power[m2_left:m2_right]) + m2_left]
m3_freq = frequency[np.argmax(power[m3_left:m3_right]) + m3_left]
print m1_freq/orbital_frequency, m2_freq/orbital_frequency , m3_freq/orbital_frequency, s1_freq/orbital_frequency, s2_freq/orbital_frequency

s1 = LombScargle(time, flux, flux_err).power(s1_freq)
s2 = LombScargle(time, flux, flux_err).power(s2_freq)

m1 = LombScargle(time, flux, flux_err).power(m1_freq)
m2 = LombScargle(time, flux, flux_err).power(m2_freq)
m3 = LombScargle(time, flux, flux_err).power(m3_freq)

print m1,m2,m3, s1, s2

s1_fit = LombScargle(time, flux + 1., flux_err).model(time, s1_freq)
s2_fit = LombScargle(time, flux + 1., flux_err).model(time, s2_freq)

residu = (flux + 1.)/s2_fit
print 'Nigga',LombScargle(time, residu, flux_err).power(s2_freq), s2
frequency2, power2 = LombScargle(time, residu, flux_err).autopower(minimum_frequency=10.**(-0.05)*orbital_frequency, maximum_frequency=60.*orbital_frequency)

time2 = np.ma.mod(time, period)

data = np.ma.column_stack((time2,flux + 1,flux_err))                     
data = data[np.lexsort((data[:,2],data[:,1],data[:,0]))]	

bins = np.linspace(0., period, 200)
digitized = np.digitize(data[:,0], bins)
bin_means = [data[:,1][digitized == i].mean() for i in range(len(bins))]
print len(bin_means), len(bins)
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,period])
ax1.grid()
ax1.set_ylim([0.96,1.03])
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data[:,0], data[:,1], 'o', ms=1)
ax1.plot(bins, bin_means, 'o', ms=5, color= 'red')

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([45.9,45.92])
ax2.set_ylim([6.6e-5,0.7])
ax2.plot(frequency/orbital_frequency, power)
ax2.set_yscale('log')
ax2.get_xaxis().get_major_formatter().set_useOffset(False)
#ax2.set_xticks([1,2,3,4,5,6])
#ax2.set_xticklabels([1,2,3,4,5,6])

data2 = np.ma.column_stack((time2,residu,flux_err))                     
data2 = data2[np.lexsort((data2[:,2],data2[:,1],data2[:,0]))]
bins2 = np.linspace(0., period, 200)
digitized2 = np.digitize(data2[:,0], bins)
bin_means2 = [data2[:,1][digitized2 == i].mean() for i in range(len(bins))]

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Flux')
ax3.set_xlim([0,period])
ax3.grid()
ax3.set_ylim([0.96,1.03])
ax3.get_yaxis().get_major_formatter().set_useOffset(False)
ax3.plot(data2[:,0], data2[:,1], 'o', ms=1)
ax3.plot(bins2, bin_means2, 'o', ms=5, color= 'red')

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_xlabel('Frequency (1/orbital period)')
ax4.set_ylabel('Power')
ax4.set_xlim([45.9,45.92])
ax4.set_ylim([6.6e-5,0.7])
ax4.plot(frequency2/orbital_frequency, power2)
ax4.set_yscale('log')
ax4.get_xaxis().get_major_formatter().set_useOffset(False)
plt.tight_layout()
plt.show()
