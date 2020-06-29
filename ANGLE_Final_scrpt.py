# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:07:45 2020

@author: asus
"""

import pandas as pd 
import numpy as np
from numpy import diff
import matplotlib.pyplot as plt
import os
import csv
from scipy.ndimage import filters
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz,medfilt
import vg


class MyList(list):
    def __init__(self, *args):
        list.__init__(self, *args)
    def indices(self, filtr=lambda x: bool(x)):
        return [i for i,x in enumerate(self) if filtr(x)]
    
def testGaussian3(A):
    b = gaussian(150, 25)
    gaV1 = filters.convolve1d(A , b/b.sum())
    #gaV2 = filters.convolve1d(B, b/b.sum())
    #gaV3 = filters.convolve1d(C, b/b.sum())
    #print(ga)
    #plt.plot(t, ga)
    return gaV1#,gaV2,gaV3


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

#SIMULTANIOUSLY READING AND EXECUTING THE DATA FOR ALL PROCESSING 
idxmap={'04':'1', '05':'2', '06':'3', '07':'4', '08':'5', '10':'6', '12':'7', '13':'8', '14':'9', '15':'10', '16':'11', '17':'12'}
#idxmap={'13':'8', '14':'9', '15':'10', '16':'11', '17':'12'}
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
    
    
            
            
G1_0=[]
G1_15=[]
G1_30=[]
SC_0=[]
SC_30=[]
SS_0=[]
SS_30=[]
for i in range(0,420,35):
  G1_0.append(mapping_list[0+i])
  G1_15.append(mapping_list[1+i])
  G1_30.append(mapping_list[2+i])
  SC_0.append(mapping_list[3+i:11+i])
  SC_30.append(mapping_list[11+i:19+i])
  SS_0.append(mapping_list[19+i:27+i])
  SS_30.append(mapping_list[27+i:35+i])
  
    
#------------------------------------------------------------------------------      
#MAPPING THE EACH FILE LIKE SUBJECT TO CONDITION TO EPOCHED DATATO TRAIL

#for each_map in mapping_list:
#    file1=each_map[0]
#    file2=each_map[1]
#file1=mapping_list[0:35][:,0]
#file2=mapping_list[0:35][:,1] 
final_avgang_SC_30 = []
for listt in SC_30[7:12]:
    print (listt)
    for each_map in listt:
        
        print (each_map)
        file1=each_map[0]
        file2=each_map[1]
        print(file1 , "    ",file2)
    
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
        Trg= position_data[:,13][0:]
        my_list = MyList(Trg)             
        indces=my_list.indices(lambda x: x>1) 
        start=indces[0]
        end=indces[-1]
        
        
        X= position_data[:,14:17][0:]
        Y= position_data[:,17:20][0:]
        Z= position_data[:,20:23][0:]
        t=np.linspace(0 ,len(X)/fs,len(X))
        #time=position_data[:,1][0:]
        X=X[start:end]
        Y=Y[start:end]
        Z=Z[start:end]
        time=t[start:end]
        vec1=np.array(X)-np.array(Y)
        vec2=np.array(Z)-np.array(Y)
        
        angleRaw=vg.angle(vec1, vec2)
        
        ANGLES=testGaussian3(angleRaw) # Window size: 250 ms or 300 samples
        ANGLES=np.asarray(ANGLES)
    #    plt.figure()
    #    plt.plot(time,ANGLES)
    #    plt.title('angle profile , gassian filter @150 size ')
    #    plt.xlabel('time(s)')
    #    plt.ylabel('angles')
    #    plt.show()
        epochs=epoch_data.to_numpy()
        epchdANG=cutEpochs(ANGLES,epochs)
    
    # saving the dataframe 
    #epchdANG.to_csv(r'D:\Thesis data\polhemus\ang1.csv', index=False, header=False)
    #    AVGANG= []
    #    for j in epchdANG:
    #        AVGANG.append(sum(j)/len(j))
    #       
        #rms=np.sqrt(np.mean(AVGANG**2))
        RMS=[]
        for k in epchdANG:
            RMS.append(np.sqrt(np.mean(k**2)))
        
        Mean=sum(RMS)/len(RMS)
        final_avgang_SC_30.append(Mean)    
#    print(Mean)    
#    plt.figure()
#    plt.plot(AVGANG)
#    plt.title('average epoched angle profile ')
#    plt.xlabel('samples')
#    plt.ylabel('angles')
#    plt.show()
        