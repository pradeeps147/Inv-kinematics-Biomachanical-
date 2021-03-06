# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:52:51 2020

@author: pradeep
"""

import pandas as pd 
import numpy
from numpy import diff
import  matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift
import pylab
#import cv2
import scipy.ndimage as ndimage
from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import scipy.optimize as op
import matplotlib.pyplot as plt
fs=1200 # Sampling rate of the signal
path = 'D:\Thesis data\polhemus'
file = 'AB_13_G1_0 - Report1.txt' 
position_data= pd.read_csv('AB_13_G1_0 - Report1.txt', sep="\t",skiprows=4)
#position_data.describe()

X= position_data.iloc[:,3][1:]
X=(X.dropna()).to_numpy()
#K=np.multiply(X,100)
time=position_data.iloc[:,2][1:]
t=numpy.linspace(0.1 ,len(X)/fs,len(X))

#def testGauss(t, X, s, fs):
#	b = gaussian(39, 10)
#	ga = filters.convolve1d(X, b/b.sum())
#	plt.plot(t, ga)
#	print ("gaerr", ssqe(ga, s, fs))
#	return ga
#

#img = ndimage.gaussian_filter(X,  order=0)
#plt.imshow(img, interpolation='nearest')
#plt.show()
#img = cv2.imread('opencv_logo.png')

#blur = cv2.Gaussian.Blur(X,(5,5))
#
#plt.subplot(121),plt.imshow(X),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
#plt.xticks([]), plt.yticks([])
#plt.show()
#def smoothListGaussian(X,degree=10):
#    window = degree*2-1
#    weight = numpy.array([1.0]*window)
#    weightGauss = []
#    for i in range(window):
#        i = i-degree+1
#        frac = i/float(window)
#        gauss = 1/(numpy.exp((4*(frac))**2))
#        weightGauss.append(gauss)
#    weight = numpy.array(weightGauss)*weight
#    smoothed = [0.0]*(len(X)-window)
#    for i in range(len(smoothed)):
#        smoothed[i] = sum(numpy.array(X[i:i+window])*weight)/sum(weight)
#    return smoothed 


#newdata=smoothListGaussian(X)



def smoothList(X, strippedXs=False, degree=25):
    if strippedXs == True:
        return strippedXs[0:-(len(X)-(len(X)-degree+1))]
    smoothed = [0]*(len(X)-degree+1)
    for i in range(len(smoothed)):
        smoothed[i] = sum(X[i:i+degree])/float(degree)
    return smoothed
smoothdata=smoothList(X, strippedXs=False, degree=25)
plt.figure()
plt.plot(t[24:],smoothdata)
plt.title()

def smoothListTriangle(X, strippedXs=False, degree=25):
    weight = []
    window = degree*2-1
    smoothed = [0.0]*(len(X)-window)
    for x in range(1, 2*degree):
        weight.append(degree-abs(degree-x))
    w = numpy.array(weight)
    for i in range(len(smoothed)):
        smoothed[i] = sum(numpy.array(X[i:i+window])*w)/float(sum(w))
    return smoothed
def smoothListGaussian(X, strippedXs=False, degree=25):
    window = degree*2-1
    weight = numpy.array([1.0]*window)
    weightGauss = []
    for i in range(window):
        i = i-degree+1
        frac = i/float(window)
        gauss = 1/(numpy.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight = numpy.array(weightGauss)*weight
    smoothed = [0.0]*(len(X)-window)
    for i in range(len(smoothed)):
        smoothed[i] = sum(numpy.array(X[i:i+window])*weight)/float(sum(weight))
    return smoothed
## DUMMY DATA ###
data = X  # [0]*30  # 30 "0"s in a row
#data[15] = 1  # the middle one is "1"
### PLOT DIFFERENT SMOOTHING FUNCTIONS ###
pylab.figure()#figsize=(550/80, 700/80)
pylab.suptitle('Data Smoothing', fontsize=15)
pylab.subplot(4, 1, 1)
p1 = pylab.plot(data )
p1 = pylab.plot(data)
a = pylab.axis()
pylab.axis([a[0], a[1], -.1, 1.1])
pylab.text(2,.3,"raw data", fontsize=14)
pylab.subplot(4, 1, 2)
p1 = pylab.plot(smoothList(data))
p1 = pylab.plot(smoothList(data))
#a = pylab.axis()
#pylab.axis([a[0], a[1], -.1, .4])
pylab.text(2, .3, "moving window average", fontsize=14)
pylab.subplot(4, 1, 3)
p1 = pylab.plot(smoothListTriangle(data))
p1 = pylab.plot(smoothListTriangle(data))
pylab.axis([a[0], a[1], -.1, .4])
pylab.text(2, .3, "moving triangle", fontsize=14)
pylab.subplot(4, 1, 4)
p1 = pylab.plot(smoothListGaussian(data))
p1 = pylab.plot(smoothListGaussian(data))
pylab.axis([a[0], a[1], -.1, .4])
pylab.text(2, .3, "moving gaussian", fontsize=14)
pylab.show()
#pylab.savefig("smooth.png", dpi=80)

from matplotlib import pyplot
series= position_data.iloc[:,3][1:]
#series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
# Tail-rolling average transform
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))
# plot original and transformed dataset
series.plot()
rolling_mean.plot(color='red')
pyplot.show()