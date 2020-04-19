# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:27:17 2020

@author: USER
"""

import pandas as pd 
import numpy as np
import  matplotlib.pyplot as plt
from scipy.fftpack import fft,fft2, fftshift
from mpl_toolkits.mplot3d import Axes3D
import sympy as sym

from scipy.signal import butter,filtfilt
#f=pd.read_table('AB_03_SS_30.txt' ,skiprows=2, sep= ' ')

fs=1200 # Sampling rate of the signal

position_data= pd.read_csv('AB_13_G1_0 - Report1.txt', sep="\t", header=None)
#position_data.describe()

X= position_data.iloc[4][1:]

t=np.linspace(0.1 ,len(X)/fs,len(X))
#freq=np.[1/t]
#print(t)
#print(X)
#z=sp.symbol('X')
#der=sym.diff(z)
#print(der)
#t=1:1/fs:len(X[1])-1
#v=[]
#for i in renge(len(X)):
    
#print(X)
#F=fft(X)
#frq=np.true_divide(1,t)
#
#plt.plot(frq,abs(F))
#velocities = []
#for pos in X:
#    velocity = float(pos/t)
#    velocities.append(velocity)

#print (velocities)

plt.plot(t,pd.to_numeric(X))
plt.title('Trajectories')
plt.xlabel('time(s)')
plt.ylabel('position(m)')
plt.show()



def smooth(data):
    #T = 1.5          # Sample Period
    fs=1200      # sample rate, Hz
    cutoff = 10     # desired cutoff frequency of the filter, Hz , 
    nyq =0.5 * fs  # Nyquist Frequency
    order = 3
    #n = int(T * fs)
    cutoff, fs, order
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

newdata=smooth(X)