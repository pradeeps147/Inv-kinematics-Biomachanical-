# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:11:48 2020

@author: pradeep
"""
import pandas as pd 
import numpy as np
from numpy import diff
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz,medfilt
import vg

path1 =  "D://Thesis data//polhemus/"
path2=   "D://Thesis data//epoched_file/"

file1 = 'AB_13_SS_30 - Report1.txt' 
file2=  'AB_5_star_30_oct2.csv'
fs=1200 # Sampling rate of the signal
nyf=fs/2
position_data= pd.read_csv((path1 + file1), sep="\t",skiprows=8)
position_data=position_data.fillna(0).to_numpy()
#position_data.describe()


#K=np.multiply(X,100)

Trg= position_data[:,24][0:]
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

X= position_data[:,15:18][0:]
Y= position_data[:,18:21][0:]
Z= position_data[:,21:24][0:]
t=np.linspace(0 ,len(X)/fs,len(X))
time=position_data[:,2][0:]
X=X[start:end]
Y=Y[start:end]
Z=Z[start:end]
time=time[start:end]
vec1=np.array(X)-np.array(Y)
vec2=np.array(Z)-np.array(Y)

angleRaw=vg.angle(vec1, vec2)
def testGaussian3(A):
    b = gaussian(150, 25)
    gaV1 = filters.convolve1d(A , b/b.sum())
    #gaV2 = filters.convolve1d(B, b/b.sum())
    #gaV3 = filters.convolve1d(C, b/b.sum())
    #print(ga)
    #plt.plot(t, ga)
    return gaV1#,gaV2,gaV3

ANGLES=testGaussian3(angleRaw) # Window size: 250 ms or 300 samples
ANGLES=np.asarray(ANGLES)
plt.figure()
plt.plot(time,ANGLES)
plt.title('angle profile')
plt.xlabel('time(s)')
plt.ylabel('angles')
plt.show()

#def testGaussian1(A):
#    b = gaussian(1200, 2)
#    gaV1 = filters.convolve1d(A , b/b.sum())
#    #gaV2 = filters.convolve1d(B, b/b.sum())
#    #gaV3 = filters.convolve1d(C, b/b.sum())
#    #print(ga)
#    #plt.plot(t, ga)
#    return gaV1#,gaV2,gaV3
#
#Vector1=testGaussian1((vec1)) # Window size: 250 ms or 300 samples
#vector1=np.asarray(Vector1)#.transpose()
#
#def testGaussian2(A):
#    b = gaussian(1200, 2)
#    gaV1 = filters.convolve1d(A , b/b.sum())
#    #gaV2 = filters.convolve1d(B, b/b.sum())
#    #gaV3 = filters.convolve1d(C, b/b.sum())
#    #print(ga)
#    #plt.plot(t, ga)
#    return gaV1#,gaV2,gaV3
#
#Vector2=testGaussian2(vec2) # Window size: 250 ms or 300 samples
#vector2=np.asarray(Vector2)#.transpose()
#
#angle=vg.angle(vector1, vector2)
#
#
#plt.figure()
#plt.plot(time,angle)
#plt.title('angle profile')
#plt.xlabel('time(s)')
#plt.ylabel('angles')
#plt.show()