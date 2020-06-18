# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:53:34 2020

@author: asus
"""

import os
import pandas as pd
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
    V1=scipy.signal.medfilt(A, 601)
    V2=scipy.signal.medfilt(B, 601)
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


#------------------------------------------------------------------------------


idxmap={'04':'1', '05':'2', '06':'3', '07':'4', '08':'5', '10':'6', '12':'7', '13':'8', '14':'9', '15':'10', '16':'11', '17':'12'}
idxcon={'G1':'randomGame', 'SC':'circle' ,'SS':'star'}

path1 =  "D:\Thesis data\polhemus\polhemus_data"
path2=   "D:\Thesis data\epoched_file"
subjectfile=[]
epochfile=[]
# r=root, d=directories, f = files
for r, d, f in os.walk(path1):
    for file in f:
        if '.txt' in file:
            subjectfile.append(os.path.join(r, file))
#
#for f in subjectfile:
#    
#    print(f)


for r, d, f in os.walk(path2):
    for file in f:
        if '.csv' in file:
           epochfile.append(os.path.join(r, file))

#for f in epochfile:
#    
#    print(f)


mapping_list=[]
for each_file in subjectfile:
    for epoch in epochfile:
        
        fid=each_file[41:].split('_')
        eid=epoch[31:].split('_')
        
#        print(fid, eid)
#        print( eid[0],"*", idxmap[fid[0]],"*" , eid[1],"*", idxcon[fid[1]] ,"*", fid[2][:-1],"*", eid[2], "*" )
#        print('\n')
        if eid[0]==idxmap[fid[0]] and eid[1]==idxcon[fid[1]] and ( fid[2][:-14]==eid[2] or fid[2][:-14]==eid[2][:-4] ) :
            
            temp=[]
            temp.append(each_file)
            temp.append(epoch)
            mapping_list.append(temp)
    
#------------------------------------------------------------------------------      


for each_map in mapping_list:
    file1=each_map[0]
    file2=each_map[1]
    
    fs=1200 # Sampling rate of the signal
    nyf=fs/2
    position_data= pd.read_csv(file1, sep="\t",skiprows=8)
    epoch_data= pd.read_csv(file2, delimiter=',', header=None)
    
    #reding the text file ,first 8 rows have file discription so remove
    #some data have NAN values so replaced by zero so that the lengh of data could not change
    position_data=position_data.fillna(0).to_numpy()
    
    
    fs=1200 # Sampling rate of the signal
    nyf=fs/2
    
    
    #time=position_data[:,1][0:]
    Trg= (position_data[:,13][0:])
    
    
    
    
    #def returnStartEnd(Trg):
    my_list = MyList(Trg)             
    indces=my_list.indices(lambda x: x>1) 
    start=indces[0]
    end=indces[-1]
        #return indces, start,end
    #index=returnStartEnd(Trg)
    
    X= position_data[:,1][0:]*100
    Y= position_data[:,2][0:]*100
    Z= position_data[:,3][0:]*100
    t=np.linspace(0 ,len(X)/fs,len(X))
    X=X[start:end]
    Y=Y[start:end]
    Z=Z[start:end]
    
    #time=time[start:end]
     
    t=t[start:end]
    
  
    
    
    M_smth=TestMedian(X,Y,Z,fs)
    M_smth=np.asarray(M_smth).transpose()
    
    
    
    pca_components=pca2(M_smth,pc_count = None)
    scalar = preprocessing.StandardScaler()
    standardized_data = scalar.fit_transform(M_smth)
    # n_components = numbers of dimenstions you want to retain
    pca = decomposition.PCA(n_components=2)
    # This line takes care of calculating co-variance matrix, eigen values, eigen vectors and multiplying top 2 eigen vectors with data-matrix X.
    pca_data = pca.fit_transform(M_smth)
    
    '''
    cumputing the velocity'''
    V_pc1=diff(pca_data[:,0])/diff(t)
    V_pc2=diff(pca_data[:,1])/diff(t)
    
    
    VM_smth=TestMedian1(V_pc1,V_pc2)
    VM_smth=np.asarray(VM_smth).transpose()
    
    '''#satrting to take the epoch data and compute the velocity profile corresponding the samples'''
    
    epochs=epoch_data.to_numpy()
    
    
    
    
    epchdV=cutEpochs(VM_smth,epochs)
    
    
    
    rms1=[]
    rms2=[]    
    for p in range(len(epchdV)):
        tempv=epchdV[p]
        rms1.append(np.sqrt(np.mean((np.asarray(tempv[:,0][0:]))**2)))
        rms2.append(np.sqrt(np.mean((np.asarray(tempv[:,1][0:]))**2)))
        
        
    AVERAGE1=sum(rms1)/len(rms1)
    AVERAGE2=sum(rms2)/len(rms2)
    
    
    
    
    AVERAGE_VELOCITY=average_velocity(AVERAGE1,AVERAGE2) 
    
        
        
        
#    plt.figure(1)
#    plt.plot(rms1)
#    plt.title('rms epoched velocity')
#    plt.xlabel('samples')
#    plt.ylabel('velocity profile 1st component(cm/s)')
#    
#    
#    plt.figure(2)
#    plt.plot(rms2) 
#    plt.title('rms epoched velocity')
#    plt.xlabel('samples')
#    plt.ylabel('velocity profile 2nd component(cm/s)')
#    plt.show()       