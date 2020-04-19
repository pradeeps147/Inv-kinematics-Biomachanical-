# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:53:53 2020

@author: asus
"""
import pandas as pd 
import numpy as np
from numpy import diff
import  matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift
import scipy
import pylab
#import cv2
import scipy.ndimage as ndimage
from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener, filtfilt, butter, gaussian,medfilt
from scipy.ndimage import filters
import scipy.optimize as op

path = 'D:\Thesis data\polhemus'
file = 'AB_13_G1_0 - Report1.txt' 
fs=1200 # Sampling rate of the signal
nyf=fs/2

position_data= pd.read_csv('AB_13_SC_30 - Report1.txt', sep="\t", skiprows=8)
position_data.fillna(0).to_numpy()
#position_data.describe()


#K=np.multiply(X,100)
time=position_data.iloc[:,2][0:]
Trg= position_data.iloc[:,24][0:].fillna(0)

#plt.figure('Raw_data')
#plt.plot(t,X)
#plt.title('Raw_data')
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')
#def ssqe(sm, X, fs):
#	return np.sqrt(np.sum(np.power(X-sm,2)))/fs
#
#for X in range(X[:,0],X[:,1],X[:,2]):
#    X.append

#def K(C1,C2,C3):
#    C1=X[:,0],C2=X[:,1],C3=X[:,2]
#    for X in len(C1,C2,C3):
#        
#        
#        return K
#def Trigger(Trg):
#    
#    if (Trg > 1).sum():
#   
#        return True
#  #print(Trg.index(element,start,end))
#    else:
#        return False
#Triggering= Trigger(Trg)
#print(Trg.index)

#def filtervalue(Trg,value):
#    for el in Trg:
#        if el.attribute>=1:
#            yield el.index
#trigg=filtervalue(Trg,1)
#print(trigg.index()) 

#[index for index,value in enumerate(Trg)] 
class MyList(list):
    def __init__(self, *args):
        list.__init__(self, *args)
    def indices(self, filtr=lambda x: bool(x)):
        return [i for i,x in enumerate(self) if filtr(x)]

#def returnStartEnd(Trg):
my_list = MyList(Trg)
indces=my_list.indices(lambda x: x>1) 
start=indces[0]
end=indces[-1]
    #return indces, start,end
#index=returnStartEnd(Trg)

X= position_data.iloc[:,3][0:]*100
Y= position_data.iloc[:,4][0:]*100
Z= position_data.iloc[:,5][0:]*100

X=(X.fillna(0)).to_numpy()
Y=(Y.fillna(0)).to_numpy()
Z=(Z.fillna(0)).to_numpy()
X=X[start:end]
Y=Y[start:end]
Z=Z[start:end]
time=time[start:end]
t=np.linspace(0.1 ,len(X)/fs,len(X))    
#subject=[]
#condition=[]
#
#for i in range(0,len(codition)):
#    for j in range(0,len(0,len(subject))):
#        
#for i in Trg:
#   if i>1:
#   return i.index()
#   else:
#   return   
     
  
#def testGauss(X,Y,Z, fs):
#    b = gaussian(300, 25)
#    gaX = filters.convolve1d(X , b/b.sum())
#    gaY = filters.convolve1d(Y, b/b.sum())
#    gaZ = filters.convolve1d(Z, b/b.sum())
#    #print(ga)
#    #plt.plot(t, ga)
#    return gaX,gaY,gaZ
#
#g_smth=testGauss(X,Y,Z,fs) # Window size: 250 ms or 300 samples
#g_smth=np.asarray(g_smth).transpose()
#plt.figure('gaussWindow')
#plt.plot(t,g_smth)
#plt.title('gaussWindow ,std=25,size=300')
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')



def TestMedian(A,B,C,fs):
    M1=scipy.signal.medfilt(A, 301)
    M2=scipy.signal.medfilt(B, 301)
    M3=scipy.signal.medfilt(C, 301)
    return M1,M2,M3
M_smth=TestMedian(X,Y,Z,fs)
M_smth=np.asarray(M_smth).transpose()
plt.figure('Mdian flter')
plt.plot(time,M_smth)
plt.title('median filter ,size=301')
plt.xlabel('time(s)')
plt.ylabel('position(cm)')

#def smooth(data):
#    #T = 1.5          # Sample Period
#    fs=1200      # sample rate, Hz
#    cutoff =10      # desired cutoff frequency of the filter, Hz , 
#    nyq =0.5 * fs  # Nyquist Frequency
#    order = 4
#    #n = int(T * fs)
#    cutoff, fs, order
#    normal_cutoff = cutoff / nyq
#    # Get the filter coefficients 
#    b, a = butter(order, normal_cutoff, btype='low', analog=False)
#    print(data)
#    y = signal.filtfilt(b, a, np.ravel(data),axis=0)
#    print(y)
#    return y , b , a
#def testGauss(t, X, fs):
#    b = gaussian(300, 25) #gaussian filter with window size 8000 and std=6
#    ga = filters.convolve1d(X, b/b.sum())
#    #print(ga)
#    #plt.plot(t, ga)
#    return ga
#
#g_smth=testGauss(t,X,fs)


    
#def testButterworth(nyf, t, X, fs):
#	b, a = butter(8, 8/nyf,btype='low', analog=False)
#	fl = filtfilt(b, a, X)
#	
#	#print (ssqe(fl, X, fs))
#	return fl
#test_butter=testButterworth(nyf,t,X,fs)
#plt.figure('lowpass_butterworth')
#plt.plot(t,test_butter)
#plt.title('lowpass_butterworth,cutoff=8Hz')
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')


#def testWiener(t, X, fs):
#	wi = wiener(X, mysize=29, noise=0.5)
#	#plt.plot(t,wi)
#	
#	return wi
#test_wiener=testWiener(t,X,fs)
#plt.figure('testwiener')
#plt.plot(t,test_wiener)
#plt.title('testwiener')
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')
#def testSpline(t, X, fs):
#	sp = UnivariateSpline(t, X, s=600)
#	#plt.plot(t,sp(t))
#	#print ("splerr", ssqe(sp(t),fs))
#	return sp(t)
#test_spline=testSpline(t,X,fs)
#plt.figure('splinefilter')
#plt.plot(t,test_spline)
#plt.title('univariate splinefilter')
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')
#
#def plotPowerSpectrum(X, w):
#	ft = np.fft.rfft(X)
#	ps = np.real(ft*np.conj(ft))*np.square(1/fs)
#	plt.plot(w, ps)
#   
##spectrum=plotPowerSpectrum(X,w)
#plt.figure('spectrum')

#    
#
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')
#smoothed=testGauss(t,X,fs)
#import numpy
#x=X
#def smooth(x,window_len=11,window='hanning'):
#    """smooth the data using a window with requested size.
#    
#    This method is based on the convolution of a scaled window with the signal.
#    The signal is prepared by introducing reflected copies of the signal 
#    (with the window size) in both ends so that transient parts are minimized
#    in the begining and end part of the output signal.
#    
#    input:
#        x: the input signal 
#        window_len: the dimension of the smoothing window; should be an odd integer
#        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#            flat window will produce a moving average smoothing.
#
#    output:
#        the smoothed signal
#        
#    example:
#
#    t=linspace(-2,2,0.1)
#    x=sin(t)+randn(len(t))*0.1
#    y=smooth(x)
#    
#    see also: 
#    
#    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#    scipy.signal.lfilter
# 
#    TODO: the window parameter could be the window itself if an array instead of a string
#    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#    """
#
##    if x.ndim != 1:
##        raise ValueError, "smooth only accepts 1 dimension arrays."
##
##    if x.size < window_len:
##        raise ValueError, "Input vector needs to be bigger than window size."
##
##
##    if window_len<3:
##        return x
##
##
##    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
##        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
##
##
##    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
##    #print(len(s))
##    if window == 'flat': #moving average
##        w=numpy.ones(window_len,'d')
##    else:
##        w=eval('numpy.'+window+'(window_len)')
##
##    y=numpy.convolve(w/w.sum(),s,mode='valid')
##    return y
#
#from numpy import *
#from pylab import *
#
#def smooth(x,window_len=11,window='hanning'):
#
##    t=linspace(-4,4,100)
#    x=X
#    xn=x  #+randn(len(t))*0.1
#    y=smooth(x)
#
#    ws=31
#
#    subplot(211)
#    plot(ones(ws))
#
#    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
#
#    hold(True)
#    for w in windows[1:]:
#        eval('plot('+w+'(ws) )')
#
#    axis([0,30,0,1.1])
#
#    legend(windows)
#    title("The smoothing windows")
#    subplot(212)
#    plot(x)
#    plot(xn)
#    for w in windows:
#        plot(smooth(xn,10,w))
#    l=['original signal', 'signal with noise']
#    l.extend(windows)
#
#    legend(l)
#    title("Smoothing a noisy signal")
#    show()
#
#
#if __name__=='__main__':
#    smooth_demo()
### Datashader (holoviews+Bokeh)
##
##import holoviews as hv
##import datashader as ds
##from holoviews.operation.datashader import aggregate, shade, datashade, dynspread
##from bokeh.models import DatetimeTickFormatter
##hv.extension('bokeh')
##
##def time_series(T = 1, N = 100, mu = 0.1, sigma = 0.1, S0 = 20):  
##    """Parameterized noisy time series"""
##    dt = float(T)/N
##    t = np.linspace(0, T, N)
##    W = np.random.standard_normal(size = N) 
##    W = np.cumsum(W)*np.sqrt(dt) # standard brownian motion
##    X = (mu-0.5*sigma**2)*t + sigma*W 
#    S = S0*np.exp(X) # geometric brownian motion
#    return S
#
#def apply_formatter(plot, element):
#    plot.handles['xaxis'].formatter = DatetimeTickFormatter()
#    
#drange = pd.date_range(start="2014-01-01", end="2016-01-01", freq='1D') # or '1min'
#dates = drange.values.astype('int64')/10**6 # Convert dates to ints
#curve = hv.Curve((dates, time_series(N=len(dates), sigma = 1)))
#%%opts RGB [finalize_hooks=[apply_formatter] width=800]
#%%opts Overlay [finalize_hooks=[apply_formatter] width=800] 
#%%opts Scatter [tools=['hover', 'box_select']] (line_color="black" fill_color="red" size=10)
#
#from holoviews.operation.timeseries import rolling, rolling_outlier_std
#smoothed = rolling(curve, rolling_window=50)
#outliers = rolling_outlier_std(curve, rolling_window=50, sigma=2)
#datashade(curve, cmap=["blue"]) * dynspread(datashade(smoothed, cmap=["red"]),max_px=1) * outliers