# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 00:45:54 2020

@author: asus
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

class MyList(list):
    def __init__(self, *args):
        list.__init__(self, *args)
    def indices(self, filtr=lambda x: bool(x)):
        return [i for i,x in enumerate(self) if filtr(x)]
    
def TestMedian(A,B,C,fs):
    M1=scipy.signal.medfilt(A, 301)
    M2=scipy.signal.medfilt(B, 301)
    M3=scipy.signal.medfilt(C, 301)
    return M1,M2,M3

def pca2(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)


def TestMedian1(A,B): 
    V1=scipy.signal.medfilt(A, 10)
    V2=scipy.signal.medfilt(B, 10)
    #M3=scipy.signal.medfilt(C, 301)
    return V1,V2

def cutEpochs(data,epochs):
    
    epoched_data=[]
    
    for i in range(len(epochs)):
        Start=int(epochs[i,0])
        End=int(epochs[i,1])
        #print(start,end)
        temp=data[Start:End]
        #T=np.linspace(0,len(temp)/fs,len(temp))
        #V=diff(temp)/diff(T)
        #print(temp)
        epoched_data.append(temp)
        
    return epoched_data


def average_velocity(num1,num2):
    return (num1+ num2)/2