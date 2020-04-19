# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:59:45 2020

@author: asus
"""

import pandas as pd 
import numpy as np
from numpy import diff
import  matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import scipy.optimize as op

fs=1200 # Sampling rate of the signal
nyf=fs/2

position_data= pd.read_csv('AB_13_SC_0 - Report1.txt', sep="\t",skiprows=5)
#position_data.describe()

X= position_data.iloc[:,5][1:]*100
X=(X.dropna()).to_numpy()
#K=np.multiply(X,100)
time=position_data.iloc[:,2][1:]

t=np.linspace(0.1 ,len(X)/fs,len(X))

V=diff(X)/diff(t)
def testGauss(t, V, fs):
    b = gaussian(8000, 6) #gaussian filter with window size 8000 and std=6
    ga = filters.convolve1d(V, b/b.sum())
    #print(ga)
    #plt.plot(t, ga)
    return ga

g_smth=testGauss(t,V,fs)
plt.figure()
plt.plot(t[:-1],g_smth)
plt.xlabel('time(s)')
plt.ylabel('Velocities(Cm/s)')
def testButterworth(nyf, t, V, fs):
	b, a = butter(8, 6/nyf,btype='low', analog=False)
	fl = filtfilt(b, a, V)
	
	#print (ssqe(fl, X, fs))
	return fl
test_butter=testButterworth(nyf,t,V,fs)
plt.figure()
plt.plot(t[:-1],test_butter)
plt.title('butterworth lowpass 6Hz')
plt.xlabel('time(s)')
plt.ylabel('Velocities(cm/s)')

plt.figure()
plt.plot(pd.to_numeric(t[:-1]),pd.to_numeric(V))
plt.title('Trajectories')
plt.xlabel('time(s)')
plt.ylabel('Velocities(Cm/s)')
plt.show()