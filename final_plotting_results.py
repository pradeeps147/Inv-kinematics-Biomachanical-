# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:18:08 2020

@author: asus
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path2 =  "D:/Thesis data/polhemus/Final_average_values/"
file2=   "final_avgvel_SC_30.csv"
g1_0= pd.read_csv((path2 + file2), delimiter=',', header=None)
#print(g1_0)
G1_0=g1_0.fillna(np.mean(g1_0)).to_numpy()
print(np.mean(G1_0))
print(np.std(G1_0))

cond_SC_30=[]
for i in range(0,96,8):
   cond_SC_30.append(np.mean(G1_0[0+i:8+i]))
   
#print(np.mean(G1_0)-np.std(G1_0)/np.sqrt(len(G1_0)))
#print((np.mean(G1_0)+np.std(G1_0)/np.sqrt(len(G1_0))))
#plt.figure()
#plt.plot(G1_0)
#plt.title('complex reaching task across all the subjects on 0 degree inclination')
#plt.xlabel('subjets')
#plt.ylabel('averaged velocity profile')
#plt.show()