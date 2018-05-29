# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:02:04 2018

@author: Ivan
"""

###############################################################################
# Import python repository from QMPL
###############################################################################
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit

plt.ioff()

# File to load
filename = r"escobar-SB-scan-1.3456MHz.csv"

# Modulation frequency in Hz
mod_f = 1.3456e6

# Estimated FWHM
FWHM_guess = 115e3

data = np.genfromtxt(filename, delimiter=',', comments='#')
                     
tdata = data[:,0]
Vdata = data[:,1]

# Normalize the time data and intensity data for improved fitting
tdata = (tdata-tdata.mean())/(tdata.max() - tdata.min())
Vdata = (Vdata - Vdata.min())/(Vdata.max() - Vdata.min())

# If there is a control voltage input, then use it
if data.shape[1] == 3:
    pzVdata = data[:,2]
# Otherwise, create mock data to not break the algorithm
elif data.shape[1] == 2:
    pzVdata = tdata.copy()

# Peak centers
peaks = []

# Fitting region
fit_region = []

# Ignored region
ignored_region = []

# Handler to get mouse click coordinates from a plot
def onclick(event, maxclicks, arr, nfig):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print("x = {0:.2e}, y = {0:.2e}".format(ix, iy))

    arr.append((ix, iy))

    if len(arr) == maxclicks:
        fig.canvas.mpl_disconnect(cid)
        plt.close(nfig)
    return

# Pick the linear voltage region
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(tdata, Vdata)

plt.xlabel("Time (a.u.)")
plt.ylabel("Control voltage (V)")
plt.title("Select the fitting range")
cid = fig.canvas.mpl_connect('button_press_event', lambda ev: onclick(ev, 2, fit_region, 1))
plt.show()

fit_region = np.array(fit_region)[:,0]
fit_region.sort()
print(fit_region)

# Now pick the peak centers
fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.plot(tdata, Vdata)

plt.xlabel("Time (a.u.)")
plt.ylabel("Detector voltage (a.u.)")
plt.title("Select the center and sideband peaks")
cid = fig.canvas.mpl_connect('button_press_event', lambda ev: onclick(ev, 3, peaks, 2))
plt.show()

peaks = np.array(peaks)
peaks = peaks[peaks[:,0].argsort()] # Sort along the first column
print(peaks)

## Now pick the ignored region boundaries
#fig = plt.figure(3)
#ax = fig.add_subplot(111)
#ax.plot(tdata, Vdata)
#
#plt.xlabel("Time (a.u.)")
#plt.ylabel("Detector voltage (a.u.)")
#plt.title("Select the bounds of the ignored region")
#cid = fig.canvas.mpl_connect('button_press_event', lambda ev: onclick(ev, 2, ignored_region, 3))
#
#plt.show()

#ignored_region = np.array(ignored_region)[:,0]
ignored_region = np.array([1,2])
ignored_region.sort()

# Create the final array on which we will be fitting (i.e. the fitting range 
# minus ignored area)
fit_ind = np.logical_and(tdata > fit_region[0], tdata < fit_region[1])
ign_ind = np.logical_not(np.logical_and(ignored_region[0] < tdata, tdata < ignored_region[1]))
indices = np.logical_and(fit_ind, ign_ind)

# Now, let's find a conversion between V and t
V_per_t, intercept, r_value, p_value, std_err = stats.linregress(tdata[indices], pzVdata[indices])

# Plot the linear fit to the piezo voltage
fig = plt.figure(3)
ax = fig.add_subplot(111)

ax.plot(tdata[indices], pzVdata[indices], ".", tdata[indices], intercept + V_per_t*tdata[indices], "-")
plt.xlabel("Time (a.u.)")
plt.ylabel("Control voltage (V)")
plt.title("Linear fit to control voltage")
plt.show()

# Plot the data that excludes the ignored region
fig = plt.figure(4)
ax = fig.add_subplot(111)
ax.plot(tdata[indices], Vdata[indices], ".")

plt.xlabel("Time (a.u.)")
plt.ylabel("Control voltage")
plt.title("Selected data")

plt.show()

## Now we define the function to be fitted on the averaged signal:
def triple_lorentzian(t, mainA, sideA, FWHM, f_per_t, center, A_offset):
    detun = (t-center)*f_per_t
    return A_offset + mainA / (1+(2*detun/FWHM)**2) + sideA * (1/(1+(2*(detun-mod_f)/FWHM)**2) + 1/(1+(2*(detun+mod_f)/FWHM)**2))

# Time to do the fitting
# First, we supply the initial guesses based on clicked results
mainA_guess = peaks[1,1]
sideA_guess = (peaks[0,1] + peaks[2,1]) / 2
center_guess = peaks[1,0]
A_offset_guess = np.min(Vdata[indices])
f_per_t_guess = 2  * mod_f / (peaks[2,0] - peaks[0,0])

guess = np.array([mainA_guess, sideA_guess, 
                  FWHM_guess, f_per_t_guess, center_guess, A_offset_guess])

# Set the scale for better fitting
scale = guess.copy()
scale[-2:] = 1
scale = np.abs(scale)

guess_n = guess/scale

# Create an aux function with normalized parameters
def aux(t, *params):
    p_n = np.array(params)
    p = p_n * scale
    return triple_lorentzian(t, *p)

popt, pcov = curve_fit(aux, tdata[indices], Vdata[indices], 
                       p0=guess_n)

popt = popt*scale

fit = lambda t: triple_lorentzian(t, *popt)

# Plot the data and the fit
fig = plt.figure(5)
ax = fig.add_subplot(111)
xdata = (tdata-popt[4])*popt[3]
ax.plot(xdata/1e6, Vdata, "-", 
        xdata/1e6, fit(tdata), "-")

plt.xlabel("Frequency (MHz)")
plt.ylabel("Control voltage")
plt.title("Fit")

plt.show()

# Finally, get some results
print("FWHM: {0:.3f} MHz".format(popt[2]/1e6))
print("Conversion: {0:.3f} MHz per Volt".format(popt[3]/V_per_t/1e6))