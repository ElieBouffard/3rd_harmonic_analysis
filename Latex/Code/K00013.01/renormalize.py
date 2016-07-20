import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

time, flux, flux_err = np.loadtxt("K00013.01.dc2test.dat", usecols = (0,1,2), unpack = True)

flux = flux + 1
transit_duration = 3.18137/24.
per = 1.763587569
time2 = np.ma.mod(time -(1.43713 - per/2.),per)

test = np.ma.masked_inside(time2, 0.75,0.8)
mean_test = flux[test.mask].mean()
print mean_test -1.                      
#time2 = np.ma.masked_inside(time2, per/2. - transit_duration, per/2. + transit_duration)
#time2 = np.ma.masked_inside(time2, per - transit_duration, per)
#time2 = np.ma.masked_inside(time2, transit_duration, 0)

#mask_tot = time2.mask
#time2 = np.ma.array(time2, mask = mask_tot)
#flux2 = flux[~time2.mask]
#flux_err2 = flux_err[~time2.mask]
#time2 = time2[~time2.mask]	

#data2 = np.ma.column_stack((time2,flux2,flux_err2))                               
#data2 = data2[np.lexsort((data2[:,2],data2[:,1],data2[:,0]))]

#new_mean = np.average(flux2)
#print new_mean
flux = flux/mean_test
test = np.column_stack((time,flux-1,flux_err))
np.savetxt('test.dat', test, fmt='%1.15E')   # use exponential notation

data = np.ma.column_stack((np.ma.mod(time -(1.43713 - per/2.),per),flux,flux_err))                               
data = data[np.lexsort((data[:,2],data[:,1],data[:,0]))]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax1.set_xlim([0,per])
ax1.grid()
#ax1.set_ylim([0.97,1.03])
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.plot(data[:,0], data[:,1], 'o', ms=1) 
plt.show()
