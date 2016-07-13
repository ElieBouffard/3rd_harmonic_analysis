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

####BOOTSTRAPPING#####
iterations = 50
fraction = 0.5
idx_list = range(len(time))
idx_list = np.asarray(idx_list)
numb_points = int(fraction*len(time))
diff = []
frequency = np.linspace(10.**(-0.05)*orbital_frequency, 10.**(0.8)*orbital_frequency, 43093)
for i in range(iterations):
	np.random.shuffle(idx_list)
	idx_list_temp = idx_list[:numb_points]
	time_temp = time[idx_list_temp]
	flux_temp = flux[idx_list_temp]
	flux_err_temp = flux_err[idx_list_temp]
	idx_list_temp2 = idx_list[numb_points:]
	time_temp2 = time[idx_list_temp2]
	flux_temp2 = flux[idx_list_temp2]
	flux_err_temp2 = flux_err[idx_list_temp2]
	#print numb_points, len(time_temp), len(flux_temp), len(flux_err_temp)
	power_temp = LombScargle(time_temp, flux_temp, flux_err_temp).power(frequency)
	power_temp2 = LombScargle(time_temp2, flux_temp2, flux_err_temp2).power(frequency)
	diff_temp = np.abs(power_temp2 - power_temp)
	diff.append(diff_temp)
	print 'iteration ', i+1, 'out of ', iterations
power = LombScargle(time, flux, flux_err).power(frequency)
diff = np.asarray(diff)
#print frequency.shape, power.shape
#print (flux_err/flux).mean()
#sigma_plus = 100. - 0.5*(100-68.3)
#sigma_minus = 100. - 0.5*(100-68.3) - 68.3
#LS_plus = np.percentile(power, sigma_plus, axis = 0)
#LS_minus = np.percentile(power, sigma_minus, axis = 0)
#LS_std = np.std(power, axis = 0)
LS_error = np.mean(diff, axis = 0)

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency])
ax2.set_ylim([6.6e-5,0.7])
#for i in range(iterations):
#ax2.loglog(frequency[i]/orbital_frequency, power[i], 'o')
#ax2.plot(frequency/orbital_frequency, LS_plus, 'o', label = 'Upper band')
#ax2.plot(frequency/orbital_frequency, LS_minus, 'o', label = 'Lower band')
ax2.plot(frequency/orbital_frequency, power, label = 'Raw power')
ax2.plot(frequency/orbital_frequency, LS_error, label = 'Error', color ='red')
ax2.legend()
ax2.get_yaxis().get_major_formatter().set_useOffset(False)
ax2.get_xaxis().get_major_formatter().set_useOffset(False)
#ax2.fill_between(frequency[0]/orbital_frequency, LS_plus, LS_minus, color = 'grey', interpolate=True)
ax2.set_yscale('log')
#ax2.set_xscale('log')
#ax2.set_xticks([1,2,3,4,5,6])
#ax2.set_xticklabels([1,2,3,4,5,6])

plt.show()
