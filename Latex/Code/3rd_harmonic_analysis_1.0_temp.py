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
LS_idx = 0                                     #0: Scypy. 1: Gatspy. 2: Astro_ML
hdulist = []
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2009166043257_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2009259160929_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2009350155506_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2010078095331_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2010174085026_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2010265121752_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2010355172524_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2011073133259_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2011177032512_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2011271113734_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2012004120508_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2012088054726_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2012179063303_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2012277125453_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2013011073258_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2013098041711_llc.fits'))
hdulist.append(fits.open('../../Data_sets/K01546.01/kplr005475431-2013131215648_llc.fits'))

planet_period = 0.917569588#0.917547
first_transit = 133.93176
transit_duration = 2.1035/24.
orbital_frequency = 1./planet_period
time = []
time_temp = []
PDCSAP_FLUX_temp = []
PDCSAP_FLUX_err_temp = []
SAP_FLUX_temp = []
SAP_FLUX_err_temp = []
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

time = np.ma.concatenate([time[0],time[1],time[2],time[3],time[4],time[5],time[6],time[7],time[8],time[9],time[10],time[11],time[12],time[13],time[14],time[15],time[16]])
PDCSAP_FLUX = np.ma.concatenate([PDCSAP_FLUX[0],PDCSAP_FLUX[1],PDCSAP_FLUX[2],PDCSAP_FLUX[3],PDCSAP_FLUX[4],PDCSAP_FLUX[5],PDCSAP_FLUX[6],PDCSAP_FLUX[7],PDCSAP_FLUX[8],PDCSAP_FLUX[9],PDCSAP_FLUX[10],PDCSAP_FLUX[11],PDCSAP_FLUX[12],PDCSAP_FLUX[13],PDCSAP_FLUX[14],PDCSAP_FLUX[15],PDCSAP_FLUX[16]])
SAP_FLUX = np.ma.concatenate([SAP_FLUX[0],SAP_FLUX[1],SAP_FLUX[2],SAP_FLUX[3],SAP_FLUX[4],SAP_FLUX[5],SAP_FLUX[6],SAP_FLUX[7],SAP_FLUX[8],SAP_FLUX[9],SAP_FLUX[10],SAP_FLUX[11],SAP_FLUX[12],SAP_FLUX[13],SAP_FLUX[14],SAP_FLUX[15],SAP_FLUX[16]])
PDCSAP_FLUX_err = np.ma.concatenate([PDCSAP_FLUX_err[0],PDCSAP_FLUX_err[1],PDCSAP_FLUX_err[2],PDCSAP_FLUX_err[3],PDCSAP_FLUX_err[4],PDCSAP_FLUX_err[5],PDCSAP_FLUX_err[6],PDCSAP_FLUX_err[7],PDCSAP_FLUX_err[8],PDCSAP_FLUX_err[9],PDCSAP_FLUX_err[10],PDCSAP_FLUX_err[11],PDCSAP_FLUX_err[12],PDCSAP_FLUX_err[13],PDCSAP_FLUX_err[14],PDCSAP_FLUX_err[15],PDCSAP_FLUX_err[16]])
time = time.astype(float)
time = np.ma.masked_invalid(time)
SAP_FLUX = SAP_FLUX.astype(float)
SAP_FLUX =  np.ma.masked_invalid(SAP_FLUX)
PDCSAP_FLUX = PDCSAP_FLUX.astype(float)
PDCSAP_FLUX = np.ma.masked_invalid(PDCSAP_FLUX)
PDCSAP_FLUX_err = np.ma.masked_invalid(PDCSAP_FLUX_err)

##Finding transits and masking them
left_boundary = (first_transit - transit_duration) % planet_period
right_boundary = (first_transit + transit_duration) % planet_period
sec_transit_left = (first_transit - planet_period/2. + transit_duration) % planet_period
sec_transit_right = (planet_period - planet_period/2. - transit_duration) % planet_period
time_temp = np.ma.mod(time,planet_period)
#time_temp = np.ma.masked_inside(time_temp, 0., right_boundary)
#time_temp = np.ma.masked_inside(time_temp, sec_transit_right, sec_transit_left)
#time_temp = np.ma.masked_inside(time_temp, left_boundary, planet_period)

filtered_data = sigma_clip(PDCSAP_FLUX, sigma=8, iters=None)
#mask_sig = np.zeros(len(time))
#chunck_pts = 5
#print len(time), int(len(time)/chunck_pts)
#for i in range(int(len(time)/chunck_pts)):	
#	filtered_data_temp = sigma_clip(PDCSAP_FLUX[chunck_pts*i:chunck_pts*(i+1)], sigma=4.4, iters=None)
#	mask_sig[chunck_pts*i:chunck_pts*(i+1)] = filtered_data_temp.mask
#	#print chunck_pts*(i+1)
#mask_sig[-(len(time) % chunck_pts):] = 1
#mask_tot = time_temp.mask + mask_sig
mask_tot = time_temp.mask + filtered_data.mask
mask_tot[mask_tot == 2] = 1
time = np.ma.array(time, mask = mask_tot)
print len(time), len(PDCSAP_FLUX)
PDCSAP_FLUX = PDCSAP_FLUX[~time.mask]
PDCSAP_FLUX_err = PDCSAP_FLUX_err[~time.mask]
time = time[~time.mask]	
fmin=10.**(-0.05)*orbital_frequency
fmax=10**(0.8)*orbital_frequency
Nf = 2000.
df = (fmax - fmin) / Nf	
f = 2.*np.pi*np.logspace(-0.05, 0.8, Nf)*orbital_frequency

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
	#print time
	power, z = lomb_scargle(time, PDCSAP_FLUX, PDCSAP_FLUX_err, f, significance = [1.-0.682689492,1.-0.999999426697])
	print z
#plt.plot(time, PDCSAP_FLUX)
#plt.show()
time = np.ma.mod(time - time.min() - 1.046,planet_period)
data = np.ma.column_stack((time,PDCSAP_FLUX))                                #2D array of t,signal
data = data[np.lexsort((data[:,1],data[:,0]))]	                          #we sort it

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,planet_period])
ax1.grid()
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data[:,0], data[:,1], 'o', ms=1)
#ax1.plot([planet_period/2.,planet_period/2.],[0.96,1.02])
#ax1.plot([planet_period/2., planet_period/2.], [data[:,1].min(), data[:,1].max()]) 

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Frequency (1/orbital period)')
ax2.set_ylabel('Power')
ax2.set_xlim([fmin/orbital_frequency,fmax/orbital_frequency])
ax2.set_ylim([6.6e-5,0.7])
print fmin,fmax
#ax2.grid()
#ax2.get_yaxis().get_major_formatter().set_useOffset(False)
if LS_idx == 0:
	#ax2.plot(f/orbital_frequency/2./np.pi, np.sqrt(4*(pgram/normval)))
	ax2.loglog(f/orbital_frequency/2./np.pi, pgram * 2. / (normval*PDCSAP_FLUX.std()**2.))
	#ax2.plot(f/orbital_frequency/2./np.pi, pgram * 2. / (normval*PDCSAP_FLUX.std()**2.))
	ax2.set_xticks([1,2,3,4,5,6])
	ax2.set_xticklabels([1,2,3,4,5,6])
	ax2.plot([fmin/2.,fmax], [z[0],z[0]])
	ax2.plot([fmin/2.,fmax], [z[1],z[1]])
	ax2.set_yticks([z[0],z[1]])
	ax2.set_yticklabels([r'$1\sigma$',r'$5\sigma$'])
elif LS_idx == 1:
	ax2.plot(f/orbital_frequency/2./np.pi, power)
elif LS_idx == 2:
	ax2.loglog(f/orbital_frequency/2./np.pi, power)
	#ax2.plot(f/orbital_frequency/2./np.pi, power)
	ax2.plot([fmin/2.,fmax], [z[0],z[0]])
	ax2.plot([fmin/2.,fmax], [z[1],z[1]])
	ax2.set_yticks([z[0],z[1]])
	ax2.set_yticklabels([r'$1\sigma$',r'$5\sigma$'])
	ax2.set_xticks([1,2,3,4,5,6])
	ax2.set_xticklabels([1,2,3,4,5,6])

plt.show()
