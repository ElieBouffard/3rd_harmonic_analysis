import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from astropy.io import fits
from astroML.time_series import lomb_scargle
from astropy.stats import sigma_clip
from astropy.stats import LombScargle

#This code computes the Lomb-Scargle periodogram of an exoplanet. First,
#load the fits files, then write the planet period, the time of its first
#transit and its duration. If the number of files is not 17, then you will
#have to change reading loop iteration number as well as the concatenate part.
#By default, the periodograms uses frequencies from 10.**(-0.05)*orbital_frequency
#to 10**(0.8)*orbital_frequency with 2000 points evenly spaced in log space.

#First, we load the .fits files.
hdulist = fits.open('../../../Data_sets/HAT-P-7/HAT7_photometry.fits')

planet_period = 2.20473080
first_transit = 2454700.813080
transit_duration = 0.1669
orbital_frequency = 1./planet_period

time = hdulist[1].data['BJD     ']
flux = hdulist[1].data['FLUX    ']

left_boundary = (first_transit - transit_duration) % planet_period
right_boundary = (first_transit + transit_duration) % planet_period
sec_transit_left = (first_transit - planet_period/2. + transit_duration) % planet_period
sec_transit_right = (first_transit - planet_period/2. - transit_duration) % planet_period
time_temp = np.ma.mod(time,planet_period)
time_temp = np.ma.masked_inside(time_temp, 0., right_boundary)
time_temp = np.ma.masked_inside(time_temp, sec_transit_right, sec_transit_left)
time_temp = np.ma.masked_inside(time_temp, left_boundary, planet_period)

print len(time)

filtered_data = sigma_clip(flux, sigma=5, iters=None)

mask_tot = time_temp.mask + filtered_data.mask
mask_tot[mask_tot == 2] = 1
time = np.ma.array(time, mask = mask_tot)

flux = flux[~time.mask]
time = time[~time.mask]

print len(time)
frequency, power = LombScargle(time, flux).autopower(minimum_frequency=10.**(-0.05)*orbital_frequency, maximum_frequency=10.**(0.8)*orbital_frequency)
print 'freq lenght:', len(frequency)

time = np.ma.mod(time - 0.9,planet_period)

data = np.ma.column_stack((time,flux))                               
data = data[np.lexsort((data[:,1],data[:,0]))]	

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,planet_period])
ax1.grid()
ax1.set_ylim([0.98,1.02])
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data[:,0], data[:,1], 'o', ms=1)

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency])
#ax2.set_ylim([6.6e-5,0.7])
ax2.loglog(frequency/orbital_frequency, power)
ax2.set_xticks([1,2,3,4,5,6])
ax2.set_xticklabels([1,2,3,4,5,6])

plt.show()
#fmin=10.**(-0.05)*orbital_frequency
#fmax=10**(0.8)*orbital_frequency
#Nf = 2000.
#df = (fmax - fmin) / Nf	
#f = 2.*np.pi*np.logspace(-0.05, 0.8, Nf)*orbital_frequency
#time = []
#time_temp = []
#PDCSAP_FLUX_temp = []
#PDCSAP_FLUX_err_temp = []
#SAP_FLUX_temp = []
#SAP_FLUX_err_temp = []

#We extract the data and for each set, we divide by the mean and we mask the invalid values
#for i in range(17):
#	time_temp.append(hdulist[i][1].data['time'])
#	tempT = np.asarray(time_temp[i])
#	tempT = tempT.astype(float)
#	tempT = np.ma.masked_invalid(tempT)
#	time_temp[i] = tempT
#	PDCSAP_FLUX_temp.append(hdulist[i][1].data['pdcsap_flux'])
#	PDCSAP_FLUX_err_temp.append(hdulist[i][1].data['pdcsap_flux_err'])
#	tempPDC = np.asarray(PDCSAP_FLUX_temp[i])
#	tempPDC = tempPDC.astype(float)
#	tempPDC = np.ma.masked_invalid(tempPDC)
#	tempPDC = tempPDC/tempPDC.mean()
#	tempPDC_err = np.asarray(PDCSAP_FLUX_err_temp[i])
#	tempPDC_err = tempPDC_err.astype(float)
#	tempPDC_err = np.ma.masked_invalid(tempPDC_err)
#	PDCSAP_FLUX_err_temp[i] = PDCSAP_FLUX_err_temp[i] / PDCSAP_FLUX_temp[i] * tempPDC 	
#	PDCSAP_FLUX_temp[i] = tempPDC
#	SAP_FLUX_temp.append(hdulist[i][1].data['sap_flux'])
#	SAP_FLUX_err_temp.append(hdulist[i][1].data['sap_flux_err'])
#	temp = np.asarray(SAP_FLUX_temp[i])
#	temp = temp.astype(float)
#	temp = np.ma.masked_invalid(temp)
#	temp = temp/temp.mean()	
#	SAP_FLUX_temp[i] = temp
#time = np.asarray(time_temp)
#SAP_FLUX = np.asarray(SAP_FLUX_temp)
#PDCSAP_FLUX = np.asarray(PDCSAP_FLUX_temp)
#PDCSAP_FLUX_err = np.asarray(PDCSAP_FLUX_err_temp)

##We combine all the sets
#time = np.ma.concatenate([time[0],time[1],time[2],time[3],time[4],time[5],time[6],time[7],time[8],time[9],time[10],time[11],time[12],time[13],time[14],time[15],time[16]])
#PDCSAP_FLUX = np.ma.concatenate([PDCSAP_FLUX[0],PDCSAP_FLUX[1],PDCSAP_FLUX[2],PDCSAP_FLUX[3],PDCSAP_FLUX[4],PDCSAP_FLUX[5],PDCSAP_FLUX[6],PDCSAP_FLUX[7],PDCSAP_FLUX[8],PDCSAP_FLUX[9],PDCSAP_FLUX[10],PDCSAP_FLUX[11],PDCSAP_FLUX[12],PDCSAP_FLUX[13],PDCSAP_FLUX[14],PDCSAP_FLUX[15],PDCSAP_FLUX[16]])
#SAP_FLUX = np.ma.concatenate([SAP_FLUX[0],SAP_FLUX[1],SAP_FLUX[2],SAP_FLUX[3],SAP_FLUX[4],SAP_FLUX[5],SAP_FLUX[6],SAP_FLUX[7],SAP_FLUX[8],SAP_FLUX[9],SAP_FLUX[10],SAP_FLUX[11],SAP_FLUX[12],SAP_FLUX[13],SAP_FLUX[14],SAP_FLUX[15],SAP_FLUX[16]])
#PDCSAP_FLUX_err = np.ma.concatenate([PDCSAP_FLUX_err[0],PDCSAP_FLUX_err[1],PDCSAP_FLUX_err[2],PDCSAP_FLUX_err[3],PDCSAP_FLUX_err[4],PDCSAP_FLUX_err[5],PDCSAP_FLUX_err[6],PDCSAP_FLUX_err[7],PDCSAP_FLUX_err[8],PDCSAP_FLUX_err[9],PDCSAP_FLUX_err[10],PDCSAP_FLUX_err[11],PDCSAP_FLUX_err[12],PDCSAP_FLUX_err[13],PDCSAP_FLUX_err[14],PDCSAP_FLUX_err[15],PDCSAP_FLUX_err[16]])

##We mask the invalid values
#time = time.astype(float)
#time = np.ma.masked_invalid(time)
#SAP_FLUX = SAP_FLUX.astype(float)
#SAP_FLUX =  np.ma.masked_invalid(SAP_FLUX)
#PDCSAP_FLUX = PDCSAP_FLUX.astype(float)
#PDCSAP_FLUX = np.ma.masked_invalid(PDCSAP_FLUX)
#PDCSAP_FLUX_err = np.ma.masked_invalid(PDCSAP_FLUX_err)

###Finding transits and masking them
#left_boundary = (first_transit - transit_duration) % planet_period
#right_boundary = (first_transit + transit_duration) % planet_period
#sec_transit_left = (first_transit - planet_period/2. + transit_duration) % planet_period
#sec_transit_right = (first_transit - planet_period/2. - transit_duration) % planet_period
#time_temp = np.ma.mod(time,planet_period)
#time_temp = np.ma.masked_inside(time_temp, 0., right_boundary)
#time_temp = np.ma.masked_inside(time_temp, sec_transit_right, sec_transit_left)
#time_temp = np.ma.masked_inside(time_temp, left_boundary, planet_period)

##We remove the outliers
#filtered_data = sigma_clip(PDCSAP_FLUX, sigma=8, iters=None)
##mask_sig = np.zeros(len(time))
##chunck_pts = 5
##print len(time), int(len(time)/chunck_pts)
##for i in range(int(len(time)/chunck_pts)):	
##	filtered_data_temp = sigma_clip(PDCSAP_FLUX[chunck_pts*i:chunck_pts*(i+1)], sigma=4.4, iters=None)
##	mask_sig[chunck_pts*i:chunck_pts*(i+1)] = filtered_data_temp.mask
##	#print chunck_pts*(i+1)
##mask_sig[-(len(time) % chunck_pts):] = 1
##mask_tot = time_temp.mask + mask_sig
#mask_tot = time_temp.mask + filtered_data.mask
#mask_tot[mask_tot == 2] = 1
#time = np.ma.array(time, mask = mask_tot)
#print len(time), len(PDCSAP_FLUX)
#PDCSAP_FLUX = PDCSAP_FLUX[~time.mask]
#PDCSAP_FLUX_err = PDCSAP_FLUX_err[~time.mask]
#time = time[~time.mask]	


#print len(time), len(PDCSAP_FLUX)

#frequency, power = LombScargle(time, PDCSAP_FLUX, PDCSAP_FLUX_err).autopower(minimum_frequency=10.**(-0.05)*orbital_frequency, maximum_frequency=10.**(0.8)*orbital_frequency)

#print 'freq lenght:', len(frequency)

##Phase folding for illustrative purposes
#time = np.ma.mod(time - time.min() - 1.046,planet_period)
#data = np.ma.column_stack((time,PDCSAP_FLUX))                                #2D array of t,signal
#data = data[np.lexsort((data[:,1],data[:,0]))]	                          #we sort it

#fig = plt.figure()
#ax1 = fig.add_subplot(2, 1, 1)
#ax1.set_xlabel('Time (days)')
#ax1.set_ylabel('Flux')
#ax1.set_xlim([0,planet_period])
#ax1.grid()
#ax1.set_ylim([0.97,1.03])
#ax1.get_yaxis().get_major_formatter().set_useOffset(False)
#ax1.plot(data[:,0], data[:,1], 'o', ms=1)
##ax1.plot([planet_period/2.,planet_period/2.],[0.96,1.02])
##ax1.plot([planet_period/2., planet_period/2.], [data[:,1].min(), data[:,1].max()]) 

#ax2 = fig.add_subplot(2, 1, 2)
#ax2.set_xlabel('Frequency (1/orbital period)')
#ax2.set_ylabel('Power')
#ax2.set_xlim([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency])
#ax2.set_ylim([6.6e-5,0.7])
#ax2.loglog(frequency/orbital_frequency, power)
##ax2.plot([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency], [z[0],z[0]])
##ax2.plot([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency], [z[1],z[1]])
##ax2.set_yticks([z[0],z[1]])
##ax2.set_yticklabels([r'$1\sigma$',r'$5\sigma$'])
#ax2.set_xticks([1,2,3,4,5,6])
#ax2.set_xticklabels([1,2,3,4,5,6])

#plt.show()
