# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:23:26 2020

@author: pradeep
"""
import pandas as pd 
import numpy as np
from numpy import diff
import  matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz,medfilt
from scipy.ndimage import filters
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

#path = 'D:\Thesis data\polhemus'
#file = 'AB_13_G1_0 - Report1.txt' 
fs=1200 # Sampling rate of the signal
nyf=fs/2
#reding the text file ,first 8 rows have file discription so removed
position_data= pd.read_csv('AB_14_SS_0 - Report1.txt', sep="\t", skiprows=8)
#some data have NAN values so replaced by zero so that the lengh of data could not change
position_data=position_data.fillna(0).to_numpy()
time=position_data[:,1][0:]
Trg= (position_data[:,23][0:])
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

X= position_data[:,2][0:]*100
Y= position_data[:,3][0:]*100
Z= position_data[:,4][0:]*100

X=X[start:end]
Y=Y[start:end]
Z=Z[start:end]

time=time[start:end]
t=np.linspace(0.1 ,len(X)/fs,len(X)) 

#def testButterworth1(nyf, t, A,B,C, fs):
#    b, a = butter(8, 8/nyf,btype='low', analog=False)
#    fl = filtfilt(b, a, A)
#    f2 = filtfilt(b, a, B)
#    f3 = filtfilt(b, a, C)
#    
#    return fl,f2,f3
#
#test_butter1=testButterworth1(nyf,t ,X,Y,Z,fs)
#test_butter1=np.asarray(test_butter1).transpose()


def TestMedian(A,B,C,fs):
    M1=scipy.signal.medfilt(A, 1201)
    M2=scipy.signal.medfilt(B, 1201)
    M3=scipy.signal.medfilt(C, 1201)
    return M1,M2,M3
M_smth=TestMedian(X,Y,Z,fs)
M_smth=np.asarray(M_smth).transpose()
#plt.figure('Mdian flter')
#plt.plot(time,M_smth)
#plt.title('median filter ,size=301')
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')

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
#
X_std = (M_smth - np.mean(M_smth,axis=0))/np.std(M_smth,axis=0)


pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_std)


def pca2(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)
pca_components=pca2(M_smth,pc_count = None)
scalar = preprocessing.StandardScaler()
standardized_data = scalar.fit_transform(M_smth)
# n_components = numbers of dimenstions you want to retain
pca = decomposition.PCA(n_components=2)
# This line takes care of calculating co-variance matrix, eigen values, eigen vectors and multiplying top 2 eigen vectors with data-matrix X.
pca_data = pca.fit_transform(M_smth)

V_PC1=diff(pca_data[:,0])/diff(time)
V_PC2=diff(pca_data[:,1])/diff(time)



plt.figure()
plt.plot(time[1:],V_PC1)
plt.title('VELOCITY ,median filter ,size=301,PCA')
plt.xlabel('time(s)')
plt.ylabel('velocity(cm/s)')

plt.figure()
plt.plot(time[1:],V_PC2)
plt.title('VELOCITY ,median filter ,size=301,PCA')
plt.xlabel('time(s)')
plt.ylabel('velocity(cm/s)')

#plt.figure()
#plt.plot(time[1:],V_PC1)
#plt.title('VELOCITY ,gaussWindow ,std=25,size=300,PCA')
#plt.xlabel('time(s)')
#plt.ylabel('velocity(cm/s)')
#
#plt.figure()
#plt.plot(time[1:],V_PC2)
#plt.title('VELOCITY ,gaussWindow ,std=25,size=300,PCA')
#plt.xlabel('time(s)')
#plt.ylabel('velocity(cm/s)')


#def testButterworth(nyf, t, A,B, fs):
#    b, a = butter(8, 8/nyf,btype='low', analog=False)
#    fl = filtfilt(b, a, A)
#    f2 = filtfilt(b, a, B)
#    
#    return fl,f2
#
#test_butter=testButterworth(nyf,t ,V_PC1,V_PC2,fs)
#test_butter=np.asarray(test_butter).transpose()


#def testGaussian(X,Y, fs):
#    b = gaussian(150, 25)
#    gaV1 = filters.convolve1d(X , b/b.sum())
#    gaV2 = filters.convolve1d(Y, b/b.sum())
#    #gaZ = filters.convolve1d(Z, b/b.sum())
#    #print(ga)
#    #plt.plot(t, ga)
#    return gaV1,gaV2
#
#V_smth=testGaussian(V_PC1,V_PC2,fs) # Window size: 250 ms or 300 samples
#V_smth=np.asarray(V_smth).transpose()
#plt.figure()
#plt.plot(t[1:],V_smth[:,0])
#plt.title('VELOCITY ,gaussWindow ,std=25,size=300,PCA')
#plt.xlabel('time(s)')
#plt.ylabel('velocity(cm/s)')
#plt.figure()
#plt.plot(t[1:],V_smth[:,1])
#plt.title('VELOCITY ,gaussWindow ,std=25,size=300,PCA')
#plt.xlabel('time(s)')
#plt.ylabel('velocity(cm/s)')





def TestMedian1(A,B,fs):
    V1=scipy.signal.medfilt(A, 1201)
    V2=scipy.signal.medfilt(B, 1201)
    #M3=scipy.signal.medfilt(C, 301)
    return V1,V2
VM_smth=TestMedian1(V_PC1,V_PC2,fs)
VM_smth=np.asarray(VM_smth).transpose()
plt.figure()
plt.plot(time[1:],VM_smth[:,0])
plt.title('VELOCITY ,median filter ,size=451,PCA')
plt.xlabel('time(s)')
plt.ylabel('velocity(cm/s)')

plt.figure()
plt.plot(time[1:],VM_smth[:,1])
plt.title('VELOCITY ,median filter ,size=451,PCA')
plt.xlabel('time(s)')
plt.ylabel('velocity(cm/s)')


