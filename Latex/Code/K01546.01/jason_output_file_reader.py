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
frequency, power = LombScargle(time, flux, flux_err).autopower(minimum_frequency=10.**(-0.05)*orbital_frequency, maximum_frequency=10.**(0.8)*orbital_frequency)

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
ax2.loglog(frequency/orbital_frequency, power)
ax2.set_xticks([1,2,3,4,5,6])
ax2.set_xticklabels([1,2,3,4,5,6])



plt.show()
