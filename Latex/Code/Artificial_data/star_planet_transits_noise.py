import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from astropy.stats import sigma_clip
from astropy.stats import LombScargle

star_amplitude = 1.5e-2
star_freq = 55.65
noise_sdev = 1.e-2

time_tot = 1000.
delta_t = 1./40
num_points = int(time_tot/delta_t)
time = np.linspace(0.,time_tot, num_points)

print num_points

star = star_amplitude*np.sin(2.*np.pi*star_freq*time) + 1.
noise = noise_sdev*np.random.randn(num_points)
signal = star + noise

planet_period = 0.917569588
planet_freq = 1./planet_period
first_transit = planet_period*0.5
transit_duration = 1.252687/24

left_boundary = (first_transit - transit_duration) % planet_period
right_boundary = (first_transit + transit_duration) % planet_period
sec_transit_right = (first_transit - planet_period/2. + transit_duration) % planet_period
sec_transit_left = (first_transit - planet_period/2. - transit_duration) % planet_period
time_temp = np.ma.mod(time,planet_period)

#####Comment these lines if you dont want to remove the transits
time_temp = np.ma.masked_inside(time_temp, left_boundary, right_boundary)
#time_temp = np.ma.masked_inside(time_temp, sec_transit_left, planet_period)
#time_temp = np.ma.masked_inside(time_temp, sec_transit_right, 0.)

time = time[~time_temp.mask]
signal = signal[~time_temp.mask]

frequency, power = LombScargle(time, signal, noise_sdev).autopower()

####Phase folding for the plots###############
offset = ((first_transit%planet_period) - planet_period/2.)
time = np.ma.mod(time - offset,planet_period)
data = np.ma.column_stack((time,signal))                                #2D array of t,signal
data = data[np.lexsort((data[:,1],data[:,0]))]	                          #we sort it

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,planet_period])
ax1.grid()
#ax1.set_ylim([0.97,1.03])
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data[:,0], data[:,1], 'o', ms=1)

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([0.,100.])
ax2.set_ylim([6.6e-5,1.])
ax2.plot(frequency/planet_freq, power)
ax2.set_yscale('log')
#ax2.set_xticks([1,2,3,4,5,6])
#ax2.set_xticklabels([1,2,3,4,5,6])
plt.show()
