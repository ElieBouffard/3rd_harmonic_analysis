import numpy as np
import numpy.ma as ma
from math import exp, log
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import sys
import time as timer
import scipy.signal as diag
from scipy.stats import genextreme as gev
from astropy.stats import sigma_clip
from astropy.stats import LombScargle

##def FAP(R_max, K, L, n, fap_levels, c, loc, scale):
#	epsilon, mu, sig = gev.fit(R_max, c = c, loc = loc, scale = scale)
#	fap_levels = 1./fap_levels 
#	#epsilon = epsilon*(-1)
#	print epsilon, mu, sig
#	return mu - sig/epsilon*(1-(-log(K*L/(fap_levels*n)))**(-epsilon))
	
star_amplitude = 1.5e-2
star_freq = np.pi/0.917569588
noise_sdev = 1.e-2


time_tot = 1000.
delta_t = 1./48
num_points = int(time_tot/delta_t)
time = np.linspace(0.,time_tot, num_points)

print num_points

planet_period = 0.917569588
planet_freq = 1./planet_period
first_transit = planet_period*0.5
transit_duration = 1.25/24

star = star_amplitude*np.sin(2.*np.pi*star_freq*time) + 1.
noise = noise_sdev*np.random.randn(num_points)
planet = star_amplitude/2.351*np.sin(2.*np.pi*planet_freq*time) 
signal = star + noise + planet

left_boundary = (first_transit - transit_duration) % planet_period
right_boundary = (first_transit + transit_duration) % planet_period
sec_transit_right = (first_transit - planet_period/2. + transit_duration) % planet_period
sec_transit_left = (first_transit - planet_period/2. - transit_duration) % planet_period
time_temp = np.ma.mod(time,planet_period)

#####Comment these lines if you dont want to remove the transits
time_temp1 = np.ma.masked_inside(time_temp, left_boundary, right_boundary)
time_temp2 = np.ma.masked_inside(time_temp, left_boundary, right_boundary)
time_temp2 = np.ma.masked_inside(time_temp2, sec_transit_left, planet_period)
time_temp2 = np.ma.masked_inside(time_temp2, sec_transit_right, 0.)
time_eclipse = np.ma.masked_inside(time_temp, sec_transit_left, planet_period)
time_eclipse = np.ma.masked_inside(time_eclipse, sec_transit_right, 0.)

time1 = time[~time_temp1.mask]
time2 = time[~time_temp2.mask]
signal1 = signal[~time_temp1.mask]
signal2 = signal[~time_temp2.mask]

print len(time), len(time1), len(time2)

frequency, power = LombScargle(time, signal, noise_sdev).autopower()
frequency1, power1 = LombScargle(time1, signal1, noise_sdev).autopower()
frequency2, power2 = LombScargle(time2, signal2, noise_sdev).autopower()


R, K, L = 500, 10, 100
fmin, fmax, n = frequency.min(), frequency.max(), len(frequency)
fmin1, fmax1, n1 = frequency1.min(), frequency1.max(), len(frequency1)
fmin2, fmax2, n2 = frequency2.min(), frequency2.max(), len(frequency2)
print len(frequency)
start_time = timer.time()
R_max, R_max1, R_max2 = np.loadtxt('R_maxs_ellipsoid.txt', unpack=True)
fap_percentage = np.percentile(R_max, 99.9) #FAP(R_max, K, L, n, 0.9, 0.02, 3.e-4, 1e-5)
fap_percentage1 = np.percentile(R_max1, 99.9) #FAP(R_max1, K, L, n, 0.9, 0.02, 3.e-4, 1e-5)
fap_percentage2 = np.percentile(R_max2, 99.9) #FAP(R_max2, K, L, n, 0.9, -1, 3.e-4, 1e-4)

print fap_percentage,fap_percentage1,fap_percentage2
plt.scatter(time2%planet_period, signal2[np.random.randint(len(time2), size=len(time2))])
#plt.plot(frequency2, LombScargle(time2, signal2[np.random.randint(N, size=N)], noise_sdev).power(frequency2))
#plt.plot([fmin2,fmax2], [fap_percentage2, fap_percentage2])
plt.show()

####Phase folding for the plots###############
offset = ((first_transit%planet_period) - planet_period/2.)
time = np.ma.mod(time - offset,planet_period)
time1 = np.ma.mod(time1 - offset,planet_period)
time2 = np.ma.mod(time2 - offset,planet_period)
data = np.ma.column_stack((time2,signal2))                                #2D array of t,signal
data1 = np.ma.column_stack((time[time_temp1.mask],signal[time_temp1.mask]))
data2 = np.ma.column_stack((time[time_eclipse.mask],signal[time_eclipse.mask]))
data = data[np.lexsort((data[:,1],data[:,0]))]	                          #we sort it
data1 = data1[np.lexsort((data1[:,1],data1[:,0]))]
data2 = data2[np.lexsort((data2[:,1],data2[:,0]))]

idx = np.asarray(range(len(time2)))
idx1 = np.asarray(range(len(time[time_temp1.mask])))
idx2 = np.asarray(range(len(time[time_eclipse.mask])))


plt.subplots_adjust(hspace=0.2)
#make outer gridspec
outer = gridspec.GridSpec(2, 1, height_ratios = [1, 3]) 
#make nested gridspecs
gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0])
gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec = outer[1], hspace = 0.1)

ax1 = plt.subplot(gs1[0])
ax1.plot()
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,planet_period])
ax1.grid()
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.scatter(data[:,0][(idx % 10.) == 0.], data[:,1][(idx % 10.) == 0.], s=1, color = 'black')   #the points common to all the cases
ax1.scatter(data1[:,0][(idx1 % 10.) == 0.], data1[:,1][(idx1 % 10.) == 0.], s=1, color = 'orange') #points in the transit
ax1.scatter(data2[:,0][(idx2 % 10.) == 0.], data2[:,1][(idx2 % 10.) == 0.], s=1, color = 'blue') #points in the eclipse

ax2 = plt.subplot(gs2[0])
ax2.set_ylabel('Power')
ax2.set_xlim([0.,13.])
ax2.set_ylim([6.6e-5,1.])
ax2.plot(frequency/planet_freq, power, color = 'black')
ax2.plot([(frequency/planet_freq).min(), (frequency/planet_freq).max()], [fap_percentage, fap_percentage], color = 'black')
ax2.set_yscale('log')

ax3 = plt.subplot(gs2[1])
ax3.set_ylabel('Power')
ax3.set_xlim([0.,13.])
ax3.set_ylim([6.6e-5,1.])
ax3.plot(frequency1/planet_freq, power1, color = 'orange')
ax3.plot([(frequency1/planet_freq).min(), (frequency1/planet_freq).max()], [fap_percentage1, fap_percentage1], color = 'orange')
ax3.set_yscale('log')

ax4 = plt.subplot(gs2[2])
ax4.set_ylabel('Power')
ax4.set_xlim([0.,13.])
ax4.set_ylim([6.6e-5,1.])
ax4.set_xlabel('Frequency (1/orbital period)')
ax4.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13])
ax4.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12,13])
ax4.plot(frequency2/planet_freq, power2, color = 'blue')
ax4.plot([(frequency2/planet_freq).min(), (frequency2/planet_freq).max()], [fap_percentage2, fap_percentage2], color = 'blue')
ax4.set_yscale('log')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.show()
