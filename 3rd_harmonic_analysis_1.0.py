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

hdulist = []
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2009166043257_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2009259160929_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2009350155506_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2010078095331_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2010174085026_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2010265121752_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2010355172524_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2011073133259_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2011177032512_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2011271113734_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2011271113734_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2012088054726_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2012179063303_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2012277125453_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2013011073258_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2013098041711_llc.fits'))
hdulist.append(fits.open('../Data_sets/K01546.01/kplr005475431-2013131215648_llc.fits'))

planet_period = 0.917569588
first_transit = 133.93176
transit_duration = 2.1035/24.
orbital_frequency = 1./planet_period
time = []
time_temp = []
PDCSAP_FLUX_temp = []
SAP_FLUX_temp = []
for i in range(17):
	time_temp.append(hdulist[i][1].data['time'])
	tempT = np.asarray(time_temp[i])
	tempT = tempT.astype(float)
	tempT = np.ma.masked_invalid(tempT)
	time_temp[i] = tempT
	PDCSAP_FLUX_temp.append(hdulist[i][1].data['pdcsap_flux'])
	tempPDC = np.asarray(PDCSAP_FLUX_temp[i])
	tempPDC = tempPDC.astype(float)
	tempPDC = np.ma.masked_invalid(tempPDC)
	tempPDC = tempPDC/tempPDC.mean()	
	PDCSAP_FLUX_temp[i] = tempPDC
	SAP_FLUX_temp.append(hdulist[i][1].data['sap_flux'])
	temp = np.asarray(SAP_FLUX_temp[i])
	temp = temp.astype(float)
	temp = np.ma.masked_invalid(temp)
	temp = temp/temp.mean()	
	SAP_FLUX_temp[i] = temp
time = np.asarray(time_temp)
SAP_FLUX = np.asarray(SAP_FLUX_temp)
PDCSAP_FLUX = np.asarray(PDCSAP_FLUX_temp)

time = np.ma.concatenate([time[0],time[1],time[2],time[3],time[4],time[5],time[6],time[7],time[8],time[9],time[10],time[11],time[12],time[13],time[14],time[15],time[16]])
PDCSAP_FLUX = np.ma.concatenate([PDCSAP_FLUX[0],PDCSAP_FLUX[1],PDCSAP_FLUX[2],PDCSAP_FLUX[3],PDCSAP_FLUX[4],PDCSAP_FLUX[5],PDCSAP_FLUX[6],PDCSAP_FLUX[7],PDCSAP_FLUX[8],PDCSAP_FLUX[9],PDCSAP_FLUX[10],PDCSAP_FLUX[11],PDCSAP_FLUX[12],PDCSAP_FLUX[13],PDCSAP_FLUX[14],PDCSAP_FLUX[15],PDCSAP_FLUX[16]])
SAP_FLUX = np.ma.concatenate([SAP_FLUX[0],SAP_FLUX[1],SAP_FLUX[2],SAP_FLUX[3],SAP_FLUX[4],SAP_FLUX[5],SAP_FLUX[6],SAP_FLUX[7],SAP_FLUX[8],SAP_FLUX[9],SAP_FLUX[10],SAP_FLUX[11],SAP_FLUX[12],SAP_FLUX[13],SAP_FLUX[14],SAP_FLUX[15],SAP_FLUX[16]])

time = time.astype(float)
time = np.ma.masked_invalid(time)
SAP_FLUX = SAP_FLUX.astype(float)
SAP_FLUX =  np.ma.masked_invalid(SAP_FLUX)
PDCSAP_FLUX = PDCSAP_FLUX.astype(float)
PDCSAP_FLUX = np.ma.masked_invalid(PDCSAP_FLUX)
#pgram = LombScargle().fit(time[~PDCSAP_FLUX.mask], PDCSAP_FLUX[~PDCSAP_FLUX.mask])
transit_number = int(2. * (time.max() - time.min()) / planet_period )
indexes = []
index_tr0 = (np.abs(time-first_transit)).argmin()
mask_transit = np.zeros(len(time))
print transit_number
for i in range(transit_number):
	pos = first_transit + i*planet_period/2.
	begining = pos - transit_duration
	begining_idx = (np.abs(time-begining)).argmin()
	end = pos + transit_duration
	end_idx = (np.abs(time-end)).argmin()
	mask_transit[begining_idx:end_idx+1] = 1
	print i+1, " out of ", transit_number, begining_idx,end_idx
if (2. * (time.max() - time.min())) % planet_period != 0.:
	pos = first_transit + (transit_number + 1)*planet_period/2.	
	begining = pos - transit_duration
	begining_idx = (np.abs(time-begining)).argmin()
	print begining_idx, len(time)
	mask_transit[begining_idx:] = 1
print first_transit - planet_period/2., time.min(), first_transit - planet_period/2. - time.min()
if (first_transit - planet_period/2.) > time.min():
	pos = first_transit - planet_period/2.
	i = 0
	while pos > time.min() :
		print i
		i = i + 1
		pos = first_transit - i*planet_period/2.
		begining = pos - transit_duration
		begining_idx = (np.abs(time-begining)).argmin()
		end = pos + transit_duration
		end_idx = (np.abs(time-end)).argmin()
		if begining < time.min():
			mask_transit[:end_idx+1] = 1
		else:
			mask_transit[begining_idx:end_idx+1] = 1
mask_tot = time.mask + mask_transit
mask_tot[mask_tot == 2] = 1 
print mask_tot
time = np.ma.array(time, mask = mask_tot)

PDCSAP_FLUX = PDCSAP_FLUX[~time.mask]
time = time[~time.mask]	
fmin=10.**(-0.05)*orbital_frequency
fmax=10.**(0.8)*orbital_frequency
Nf = 2000.
df = (fmax - fmin) / Nf	
f = np.linspace(fmin, fmax, Nf)
normval = len(time)
print len(time), len(PDCSAP_FLUX)
print np.where(time == 133.31024891707057)[0][0], np.where(time == 1589.3456880728409)[0][0]
#pgram = LombScargle(time, PDCSAP_FLUX)
#power = pgram.score(1./f)
time = time - time.min()
plt.plot(time, PDCSAP_FLUX)
plt.show()
time = np.ma.mod(time + planet_period/2. - 0.586,planet_period)                                           #Phase-folding
data = np.ma.column_stack((time,PDCSAP_FLUX))                                #2D array of t,signal
data = data[np.lexsort((data[:,1],data[:,0]))]	                          #we sort it

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,planet_period])
ax1.grid()
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data[:,0], data[:,1], 'o', ms=1)
#ax1.plot([planet_period/2., planet_period/2.], [data[:,1].min(), data[:,1].max()]) 

#ax2 = fig.add_subplot(2, 1, 2)
#ax2.set_xlabel('Frequency (1/orbital period)')
#ax2.set_ylabel('Power')
#ax2.grid()
#ax2.get_yaxis().get_major_formatter().set_useOffset(False)
#ax2.plot(f/orbital_frequency, np.sqrt(4*(pgram/normval)), 'o')
#ax2.set_xlim([fmin,fmax])
plt.show()
