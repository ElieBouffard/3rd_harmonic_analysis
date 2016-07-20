import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from astropy.stats import sigma_clip
from astropy.stats import LombScargle

####We start by opening the files and extracting the data#####
parameters = np.loadtxt("../../Jason_code_outputs/n0.dat", usecols = ([1]))
transit_center = parameters[8]
period = parameters[9]
orbital_frequency = 1./period
time, flux, flux_err = np.loadtxt("../../Jason_code_outputs/newfile.dat", usecols = (0,1,2), unpack = True)

print transit_center, period

####MONTE-CARLO STUFF BRO#####
iterations = 50
frequency = []
power = []
for i in range(iterations):
	flux_temp = flux + flux_err*np.random.randn(len(flux_err))
	frequency_temp, power_temp = LombScargle(time, flux_temp, flux_err).autopower(minimum_frequency=10**(-0.05)*orbital_frequency, maximum_frequency=10**(0.8)*orbital_frequency)
	frequency.append(frequency_temp)
	power.append(power_temp)
	print 'iteration ', i, 'out of ', iterations - 1
frequency = np.asarray(frequency)
power = np.asarray(power)
print frequency.shape, power.shape
print (flux_err/flux).mean()
sigma_plus = 100. - 0.5*(100-68.3)
sigma_minus = 100. - 0.5*(100-68.3) - 68.3
LS_plus = np.percentile(power, sigma_plus, axis = 0)
LS_minus = np.percentile(power, sigma_minus, axis = 0)
LS_std = np.std(power, axis = 0)
#print (LS_plus/LS_minus * 100).mean(), (LS_plus/LS_minus * 100).std(), (LS_plus/LS_minus * 100).min(), frequency[0][(LS_plus/LS_minus * 100).argmin()]/orbital_frequency, (LS_plus/LS_minus * 100).max(), frequency[0][(LS_plus/LS_minus * 100).argmax()]/orbital_frequency
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency])
ax2.set_ylim([6.6e-5,0.7])
#for i in range(iterations):
#	ax2.loglog(frequency[i]/orbital_frequency, power[i], 'o')
ax2.plot(frequency[0]/orbital_frequency, LS_plus, 'o', label = 'Upper band')
ax2.plot(frequency[0]/orbital_frequency, LS_minus, 'o', label = 'Lower band')
ax2.plot(frequency[0]/orbital_frequency, LS_std, label = 'Stdev')
ax2.legend()
ax2.get_yaxis().get_major_formatter().set_useOffset(False)
ax2.get_xaxis().get_major_formatter().set_useOffset(False)
#ax2.fill_between(frequency[0]/orbital_frequency, LS_plus, LS_minus, color = 'grey', interpolate=True)
#ax2.set_yscale('log')
#ax2.set_xscale('log')
#ax2.set_xticks([1,2,3,4,5,6])
#ax2.set_xticklabels([1,2,3,4,5,6])

plt.show()
#data = np.ma.column_stack((time2,flux,flux_err))                     
#data = data[np.lexsort((data[:,2],data[:,1],data[:,0]))]	

#fig = plt.figure()
#ax1 = fig.add_subplot(2, 1, 1)
#ax1.set_xlabel('Time (days)')
#ax1.set_ylabel('Flux')
#ax1.set_xlim([0,period])
#ax1.grid()
##ax1.set_ylim([0.97,1.03])
#ax1.get_yaxis().get_major_formatter().set_useOffset(False)
#ax1.plot(data[:,0], data[:,1], 'o', ms=1)

#ax2 = fig.add_subplot(2, 1, 2)
#ax2.set_xlabel('Frequency (1/orbital period)')
#ax2.set_ylabel('Power')
#ax2.set_xlim([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency])
#ax2.set_ylim([6.6e-5,0.7])
#ax2.loglog(frequency/orbital_frequency, power)
#ax2.set_xticks([1,2,3,4,5,6, 44, 88])
#ax2.set_xticklabels([1,2,3,4,5,6, 44, 88])

#plt.show()
