import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sys
import scipy.signal as diag
from scipy.optimize import curve_fit
from astropy.io import fits
from astroML.time_series import lomb_scargle
from astropy.stats import sigma_clip
from astropy.stats import LombScargle

#This code computes the Lomb-Scargle periodogram of an exoplanet. First,
#load the fits files, then write the planet period, the time of its first
#transit and its duration. If the number of files is not 17, then you will
#have to change reading loop iteration number as well as the concatenate part.
#This version fits a simple trapezoid to the phase-folded data.

hdulist = []

#First, we load the .fits files.
#### write the number of fits files 
files_number = 17
####
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
for i in range(files_number):
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
########################################################################################
###IF THE NUMBER OF FILES ISNT 17, YOU NEED TO CHANGE THE NEXT 4 LINES ACCORDINGLY!!####
########################################################################################
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

print np.sum(PDCSAP_FLUX_err.mask), np.sum(PDCSAP_FLUX.mask), np.sum(time.mask)

#We remove the outliers
filtered_data = sigma_clip(PDCSAP_FLUX, sigma=8, iters=None)

####Fitting the transit on the phase-folded data####
def transit_ps_fit(time, transit_center, transit_duration, transit_depth, ingress_duration):
	flux = 0.0*time +1.0
	slope = transit_depth/ingress_duration
	flux[time < (transit_center - transit_duration/2. - ingress_duration/2.)] = 1.0
	flux[(time >= (transit_center - transit_duration/2. - ingress_duration/2.)) & (time <= (transit_center - transit_duration/2 + ingress_duration/2.))] = 1.0 - slope*(time[(time >= (transit_center - transit_duration/2. - ingress_duration/2.)) & (time <= (transit_center - transit_duration/2 + ingress_duration/2.))] - (transit_center - transit_duration/2. - ingress_duration/2.))
	flux[(time > (transit_center - transit_duration/2 + ingress_duration/2.)) & (time < (transit_center + transit_duration/2 - ingress_duration/2.))] = 1.0 - transit_depth
	flux[(time >= (transit_center + transit_duration/2 - ingress_duration/2.)) & (time <= (transit_center + transit_duration/2 + ingress_duration/2.))] = 1.0 - transit_depth + slope*(time[(time >= (transit_center + transit_duration/2 - ingress_duration/2.)) & (time <= (transit_center + transit_duration/2 + ingress_duration/2.))] - (transit_center + transit_duration/2. - ingress_duration/2.))
	flux[time > (transit_center + transit_duration/2 + ingress_duration/2.)] = 1.0
	return flux

mask_tot = time.mask + filtered_data.mask
mask_tot[mask_tot == 2] = 1
print np.sum(mask_tot)
time = np.ma.array(time, mask = mask_tot)
print len(time), len(PDCSAP_FLUX), len(PDCSAP_FLUX_err)
PDCSAP_FLUX = PDCSAP_FLUX[~time.mask]
PDCSAP_FLUX_err = PDCSAP_FLUX_err[~time.mask]
time = time[~time.mask]	
print len(time), len(PDCSAP_FLUX), len(PDCSAP_FLUX_err)
offset = ((first_transit%planet_period) - planet_period/2.)
time2 = np.ma.mod(time - offset,planet_period)
data = np.ma.column_stack((time2,PDCSAP_FLUX,PDCSAP_FLUX_err))                                #2D array of t,signal
data = data[np.lexsort((data[:,2],data[:,1],data[:,0]))]
time_dat = data[:,0]
PDC_dat = data[:,1]
PDC_err_dat = data[:,2]
print all(time_dat[i] <= time_dat[i+1] for i in xrange(len(time_dat)-1))
print np.sum(time_dat.mask), np.sum(PDC_dat.mask), np.sum(PDC_err_dat.mask)
popt, pcov = curve_fit(transit_ps_fit, time_dat, PDC_dat, p0=([planet_period/2., 2./24., 0.02, 0.02]), sigma = PDC_err_dat) 
perr = np.sqrt(np.diag(pcov))
print 'transit center:', popt[0], 'plus or minus', perr[0]
print 'transit_duration:', popt[1], 'plus or minus', perr[1]
print 'transit_depth:', popt[2], 'plus or minus', perr[2]
print 'ingress_duration:', popt[3], 'plus or minus', perr[3]


###Now, let's divide the data by the transit model###
transit_number = int((time.max() - first_transit)/planet_period)
new_data = PDCSAP_FLUX
new_data_err = PDCSAP_FLUX_err
for i in range(transit_number):
	if i == 0:
		flux_fit = transit_ps_fit(time[(time <= (first_transit + 2.*popt[1]))], first_transit, popt[1], popt[2], popt[3])
		new_data[(time <= (first_transit + 2.*popt[1]))] = PDCSAP_FLUX[(time <= (first_transit + 2.*popt[1]))]/flux_fit
		
	elif i == (transit_number - 1):
		flux_fit = transit_ps_fit(time[(time >= (i*planet_period + first_transit - 2.*popt[1]))], i*planet_period + first_transit, popt[1], popt[2], popt[3])
		new_data[(time >= (i*planet_period + first_transit - 2.*popt[1]))] = PDCSAP_FLUX[(time >= (i*planet_period + first_transit - 2.*popt[1]))]/flux_fit
	else:
		flux_fit = transit_ps_fit(time[(time >= (i*planet_period + first_transit - 2.*popt[1])) & (time <= (i*planet_period + first_transit + 2.*popt[1]))], i*planet_period + first_transit, popt[1], popt[2], popt[3])
		new_data[(time >= (i*planet_period + first_transit - 2.*popt[1])) & (time <= (i*planet_period + first_transit + 2.*popt[1]))] = PDCSAP_FLUX[(time >= (i*planet_period + first_transit - 2.*popt[1])) & (time <= (i*planet_period + first_transit + 2.*popt[1]))]/flux_fit
	

frequency, power = LombScargle(time, new_data, PDCSAP_FLUX_err).autopower(minimum_frequency=10.**(-0.05)*orbital_frequency, maximum_frequency=10.**(0.8)*orbital_frequency)

data2 = np.ma.column_stack((time2,new_data,PDCSAP_FLUX_err))                                #2D array of t,signal
data2 = data2[np.lexsort((data2[:,2],data2[:,1],data2[:,0]))]	


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,planet_period])
ax1.grid()
ax1.set_ylim([0.97,1.03])
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data2[:,0], data2[:,1], 'o', ms=1)

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([frequency.min()/orbital_frequency,frequency.max()/orbital_frequency])
ax2.set_ylim([6.6e-5,0.7])
ax2.loglog(frequency/orbital_frequency, power)
ax2.set_xticks([1,2,3,4,5,6])
ax2.set_xticklabels([1,2,3,4,5,6])

plt.show()
