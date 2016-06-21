import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from astropy import constants as const
from gatspy.periodic import LombScargle
from gatspy.periodic import LombScargleFast
from astropy.io import fits
from astroML.time_series import lomb_scargle
from astropy.stats import sigma_clip
from astropy.modeling.models import Lorentz1D
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
LS_idx = 2                                     #0: Scypy. 1: Gatspy. 2: Astro_ML
hdulist = []

#First, we load the .fits files.
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2009166043257_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2009259160929_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2009350155506_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2010078095331_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2010174085026_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2010265121752_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2010355172524_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2011073133259_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2011177032512_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2011271113734_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2012004120508_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2012088054726_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2012179063303_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2012277125453_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2013011073258_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2013098041711_llc.fits'))
hdulist.append(fits.open('../../../Data_sets/K01546.01/kplr005475431-2013131215648_llc.fits'))

planet_period = 0.917569588#0.917547#
first_transit = 133.93176
transit_duration = 2.1035/24.
orbital_frequency = 1./planet_period
time = []
time_temp = []
PDCSAP_FLUX_temp = []
PDCSAP_FLUX_err_temp = []
SAP_FLUX_temp = []
SAP_FLUX_err_temp = []
#We extract the data and for each set, we divide by the mean and we mask the invalid values
for i in range(17):
	time_temp.append(hdulist[i][1].data['time'])
	tempT = np.asarray(time_temp[i])
	tempT = tempT.astype(float)
	tempT = np.ma.masked_invalid(tempT)
	time_temp[i] = tempT
	PDCSAP_FLUX_temp.append(hdulist[i][1].data['pdcsap_flux'])
	PDCSAP_FLUX_err_temp.append(hdulist[i][1].data['pdcsap_flux_err'])
	tempPDC = np.asarray(PDCSAP_FLUX_temp[i])
	tempPDC = tempPDC.astype(float)
	tempPDC = np.ma.masked_invalid(tempPDC)
	tempPDC = tempPDC/tempPDC.mean()
	tempPDC_err = np.asarray(PDCSAP_FLUX_err_temp[i])
	tempPDC_err = tempPDC_err.astype(float)
	tempPDC_err = np.ma.masked_invalid(tempPDC_err)
	PDCSAP_FLUX_err_temp[i] = PDCSAP_FLUX_err_temp[i] / PDCSAP_FLUX_temp[i] * tempPDC 	
	PDCSAP_FLUX_temp[i] = tempPDC
	SAP_FLUX_temp.append(hdulist[i][1].data['sap_flux'])
	SAP_FLUX_err_temp.append(hdulist[i][1].data['sap_flux_err'])
	temp = np.asarray(SAP_FLUX_temp[i])
	temp = temp.astype(float)
	temp = np.ma.masked_invalid(temp)
	temp = temp/temp.mean()	
	SAP_FLUX_temp[i] = temp
time = np.asarray(time_temp)
SAP_FLUX = np.asarray(SAP_FLUX_temp)
PDCSAP_FLUX = np.asarray(PDCSAP_FLUX_temp)
PDCSAP_FLUX_err = np.asarray(PDCSAP_FLUX_err_temp)

#We combine all the sets

time = np.ma.concatenate([time[0],time[1],time[2],time[3],time[4],time[5],time[6],time[7],time[8],time[9],time[10],time[11],time[12],time[13],time[14],time[15],time[16]])
PDCSAP_FLUX = np.ma.concatenate([PDCSAP_FLUX[0],PDCSAP_FLUX[1],PDCSAP_FLUX[2],PDCSAP_FLUX[3],PDCSAP_FLUX[4],PDCSAP_FLUX[5],PDCSAP_FLUX[6],PDCSAP_FLUX[7],PDCSAP_FLUX[8],PDCSAP_FLUX[9],PDCSAP_FLUX[10],PDCSAP_FLUX[11],PDCSAP_FLUX[12],PDCSAP_FLUX[13],PDCSAP_FLUX[14],PDCSAP_FLUX[15],PDCSAP_FLUX[16]])
SAP_FLUX = np.ma.concatenate([SAP_FLUX[0],SAP_FLUX[1],SAP_FLUX[2],SAP_FLUX[3],SAP_FLUX[4],SAP_FLUX[5],SAP_FLUX[6],SAP_FLUX[7],SAP_FLUX[8],SAP_FLUX[9],SAP_FLUX[10],SAP_FLUX[11],SAP_FLUX[12],SAP_FLUX[13],SAP_FLUX[14],SAP_FLUX[15],SAP_FLUX[16]])
PDCSAP_FLUX_err = np.ma.concatenate([PDCSAP_FLUX_err[0],PDCSAP_FLUX_err[1],PDCSAP_FLUX_err[2],PDCSAP_FLUX_err[3],PDCSAP_FLUX_err[4],PDCSAP_FLUX_err[5],PDCSAP_FLUX_err[6],PDCSAP_FLUX_err[7],PDCSAP_FLUX_err[8],PDCSAP_FLUX_err[9],PDCSAP_FLUX_err[10],PDCSAP_FLUX_err[11],PDCSAP_FLUX_err[12],PDCSAP_FLUX_err[13],PDCSAP_FLUX_err[14],PDCSAP_FLUX_err[15],PDCSAP_FLUX_err[16]])

#We mask the invalid values

time = time.astype(float)
time = np.ma.masked_invalid(time)
SAP_FLUX = SAP_FLUX.astype(float)
SAP_FLUX =  np.ma.masked_invalid(SAP_FLUX)
PDCSAP_FLUX = PDCSAP_FLUX.astype(float)
PDCSAP_FLUX = np.ma.masked_invalid(PDCSAP_FLUX)
PDCSAP_FLUX_err = np.ma.masked_invalid(PDCSAP_FLUX_err)

##Finding transits and masking them




#mask_sig = np.zeros(len(time))
#chunck_pts = 5
#print len(time), int(len(time)/chunck_pts)
#for i in range(int(len(time)/chunck_pts)):	
#	filtered_data_temp = sigma_clip(PDCSAP_FLUX[chunck_pts*i:chunck_pts*(i+1)], sigma=4.4, iters=None)
#	mask_sig[chunck_pts*i:chunck_pts*(i+1)] = filtered_data_temp.mask
#	#print chunck_pts*(i+1)
#mask_sig[-(len(time) % chunck_pts):] = 1
#mask_tot = time_temp.mask + mask_sig

#We find the global mask

#We remove the masked values




fig, axes = plt.subplots(4, 3)#, sharey = 'all')
axes = axes.ravel()
fig.subplots_adjust(hspace=0, wspace=0)
print len(time), len(PDCSAP_FLUX)
if LS_idx == 0:
	normval = len(time)
	pgram = diag.lombscargle(time, PDCSAP_FLUX - PDCSAP_FLUX.mean(), f)
	power, z = lomb_scargle(time, PDCSAP_FLUX, PDCSAP_FLUX_err, f, significance = [1.-0.682689492,1.-0.999999426697])
	print z
elif LS_idx == 1:
	pgram = LombScargle(time, PDCSAP_FLUX)
	power = pgram.score(2.*np.pi/f)
elif LS_idx == 2:
	for i in range(6):
		if i < 3:
			planet_period = 0.917569588
		else:
			planet_period = 0.917547
		orbital_frequency = 1./planet_period
		fmin=10.**(-0.05)*orbital_frequency
		fmax=10**(0.8)*orbital_frequency
		f = 2.*np.pi*np.logspace(-0.05, 0.8, 2000)*orbital_frequency
		left_boundary = (first_transit - transit_duration) % planet_period
		right_boundary = (first_transit + transit_duration) % planet_period
		sec_transit_left = (first_transit - planet_period/2. + transit_duration) % planet_period
		sec_transit_right = (first_transit - planet_period/2. - transit_duration) % planet_period
		time_temp = np.ma.mod(time,planet_period)
		#####Comment these lines if you dont want to remove the transits
		if i % 3 == 1:
			time_temp = np.ma.masked_inside(time_temp, 0., right_boundary)
			time_temp = np.ma.masked_inside(time_temp, left_boundary, planet_period)
		elif i % 3 == 2:
			time_temp = np.ma.masked_inside(time_temp, sec_transit_right, sec_transit_left)
			time_temp = np.ma.masked_inside(time_temp, 0., right_boundary)
			time_temp = np.ma.masked_inside(time_temp, left_boundary, planet_period)
		filtered_data = sigma_clip(PDCSAP_FLUX, sigma=8, iters=None)
		mask_tot = time_temp.mask + filtered_data.mask
		mask_tot[mask_tot == 2] = 1
		time1 = np.ma.array(time, mask = mask_tot)
		PDCSAP_FLUX_temp = PDCSAP_FLUX[~time1.mask]
		PDCSAP_FLUX_err_temp = PDCSAP_FLUX_err[~time1.mask]
		time_temp = time[~time1.mask]	
		power, z = lomb_scargle(time_temp, PDCSAP_FLUX_temp, PDCSAP_FLUX_err_temp, f, significance = [1.-0.682689492,1.-0.999999426697])
		print z
		time_2 = np.ma.mod(time_temp - time_temp.min() - 1.046,planet_period)
		data = np.ma.column_stack((time_2,PDCSAP_FLUX_temp))                           
		data = data[np.lexsort((data[:,1],data[:,0]))]	
		#bins = np.linspace(0.,planet_period,1000)
		#bins2 = np.linspace(planet_period/2 +(1.e-06),planet_period,5000)
		#bins = np.concatenate([bins, bins2])
		#digitized = np.digitize(data[:,0],bins)
		#print digitized
		#bin_means_t = []
		#bin_means_sign = []
		#for j in range(1, len(bins)):
		#	bin_means_t.append(data[:,0][digitized == j].mean())
		#	bin_means_sign.append(data[:,1][digitized == j].mean()) 
			#print i
		#bin_means_t = np.ma.asarray(bin_means_t)
		#bin_means_sign = np.ma.asarray(bin_means_sign)
		if i < 3:                      
			axes[i].set_xlim([0,planet_period])
			axes[i].set_ylim([0.96,1.04])
			#axes[i].get_yaxis().get_major_formatter().set_useOffset(False)
			axes[i].plot(data[:,0],data[:,1], 'o', ms=1)
			axes[i+3].loglog(f/orbital_frequency/2./np.pi, power)
			axes[i+3].set_ylim([6.6e-5,0.7])
			axes[i+3].plot([fmin/2.,fmax], [z[0],z[0]])
			axes[i+3].plot([fmin/2.,fmax], [z[1],z[1]])
			axes[i+3].set_xlim([fmin/orbital_frequency,fmax/orbital_frequency])
		else:
			axes[i+3].set_xlim([0,planet_period])
			axes[i+3].set_ylim([0.96,1.04])
			#axes[i+3].get_yaxis().get_major_formatter().set_useOffset(False)
			axes[i+3].plot(data[:,0], data[:,1], 'o', ms=1)
			axes[i+6].loglog(f/orbital_frequency/2./np.pi, power)
			axes[i+6].set_ylim([6.6e-5,0.7])
			axes[i+6].plot([fmin/2.,fmax], [z[0],z[0]])
			axes[i+6].plot([fmin/2.,fmax], [z[1],z[1]])
			axes[i+6].set_xlim([fmin/orbital_frequency,fmax/orbital_frequency])
			
		if i%3==0:
			if i == 0:
				axes[i].set_xlabel('Time (days)')
				axes[i].set_ylabel('Flux')
				axes[i].set_xticks([])
				axes[i+3].set_ylabel('Power')
				axes[i+3].set_yticks([z[0],z[1]])
				axes[i+3].set_yticklabels([r'$1\sigma$',r'$5\sigma$'])
			if i == 3:
				axes[i+3].set_xlabel('Time (days)')
				axes[i+3].set_ylabel('Flux')
				axes[i+3].set_xticks([])
				axes[i+6].set_ylabel('Power')
				axes[i+6].set_yticks([z[0],z[1]])
				axes[i+6].set_yticklabels([r'$1\sigma$',r'$5\sigma$'])	
				axes[i+6].set_xticks([1.,2.,3.,4.,5.,6.])
				axes[i+6].set_xticklabels([1,2,3,4,5,6])
		else:
			if i < 3:
				axes[i+3].set_yticks([])
				axes[i].set_yticks([])
				axes[i].set_xticks([])	
				axes[i+3].set_xticks([1.,2.,3.,4.,5.,6.])
			else:
				axes[i+3].set_yticks([])
				axes[i+3].set_xticks([])
				axes[i+6].set_yticks([])
				axes[i+6].set_xticks([1.,2.,3.,4.,5.,6.])
				axes[i+6].set_xticklabels([1,2,3,4,5,6])
				if i == 4:
					axes[i+6].set_xlabel('Frequency (1/orbital period)')
plt.show()
