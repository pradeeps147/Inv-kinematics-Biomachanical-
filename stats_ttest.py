# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:28:02 2020

@author: asus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
from scipy import stats
import seaborn as sns



path= "D:/Thesis data/polhemus/Final_average_values/final_stats_data/" 
file1="cond_G1_0.csv"
file2="cond_G1_15.csv"
file3="cond_G1_30.csv"
file4="cond_SC_0.csv"
file5="cond_SC_30.csv"
file6="cond_SS_0.csv"
file7="cond_SS_30.csv"
file1_a="condavg_G1_0.csv"
file2_a="condavg_G1_15.csv"
file3_a="condavg_G1_30.csv"
file4_a="condavg_SC_0.csv"
file5_a="condavg_SC_30.csv"
file6_a="condavg_SS_0.csv"
file7_a="condavg_SS_30.csv"
G1_0 = pd.read_csv((path + file1), delimiter=',', header=None)
G1_0 = G1_0.fillna(np.mean(G1_0)).to_numpy()
G1_15 = pd.read_csv((path + file2), delimiter=',', header=None)
G1_15 = G1_15.fillna(np.mean(G1_15)).to_numpy()
G1_30 = pd.read_csv((path + file3), delimiter=',', header=None)
G1_30 = G1_30.fillna(np.mean(G1_30)).to_numpy()
SC_0 = pd.read_csv((path + file4), delimiter=',', header=None)
SC_0 = SC_0.fillna(np.mean(SC_0)).to_numpy()
SC_30 = pd.read_csv((path + file5), delimiter=',', header=None)
SC_30 = SC_30.fillna(np.mean(SC_30)).to_numpy()
SS_0 = pd.read_csv((path + file6), delimiter=',', header=None)
SS_0 = SS_0.fillna(np.mean(SS_0)).to_numpy()
SS_30 = pd.read_csv((path + file7), delimiter=',', header=None)
SS_30 = SS_30.fillna(np.mean(SS_30)).to_numpy()

G1_0_a = pd.read_csv((path + file1_a), delimiter=',', header=None)
G1_0_a = G1_0_a.fillna(np.mean(G1_0_a)).to_numpy()
G1_15_a = pd.read_csv((path + file2_a), delimiter=',', header=None)
G1_15_a = G1_15_a.fillna(np.mean(G1_15_a)).to_numpy()
G1_30_a = pd.read_csv((path + file3_a), delimiter=',', header=None)
G1_30_a = G1_30_a.fillna(np.mean(G1_30_a)).to_numpy()
SC_0_a = pd.read_csv((path + file4_a), delimiter=',', header=None)
SC_0_a = SC_0_a.fillna(np.mean(SC_0_a)).to_numpy()
SC_30_a = pd.read_csv((path + file5_a), delimiter=',', header=None)
SC_30_a = SC_30_a.fillna(np.mean(SC_30_a)).to_numpy()
SS_0_a = pd.read_csv((path + file6_a), delimiter=',', header=None)
SS_0_a = SS_0_a.fillna(np.mean(SS_0_a)).to_numpy()
SS_30_a = pd.read_csv((path + file7_a), delimiter=',', header=None)
SS_30_a = SS_30_a.fillna(np.mean(SS_30_a)).to_numpy()


complex_g =(G1_0_a + G1_0_a+ G1_30_a)/3
simple_s = (SC_0_a + SS_0_a + SC_30_a + SS_30_a)/4
#simple_S_30_a= (SC_30_a + SS_30_a)/2
inclination_0_a=(G1_0_a + SC_0_a + SS_0_a)/3
inclination_30_a=(G1_30_a + SC_30_a + SS_30_a)/3
#complex_vs_simple_0_a=stats.ttest_ind(complex_g,simple_s)
#complex_vs_simple_30_a=stats.ttest_ind(complex_G_a,simple_S_30_a)
inclinations_0_vs_30_a=stats.ttest_ind(inclination_0_a,inclination_30_a)
simple_vs_complex =stats.ttest_ind(simple_s , complex_g)
#print(complex_vs_simple_0_a)
#print(complex_vs_simple_30_a)
#print(inclinations_0_vs_30_a)
#print()

complex_G =(G1_0 + G1_15+ G1_30)/3

simple_S= (SC_30 + SS_30+ SC_0 + SS_0)/4
inclination_0=(G1_0 + SC_0 + SS_0)/3
inclination_30=(G1_30 + SC_30 + SS_30)/3
complex_vs_simple_0=stats.ttest_ind(complex_G,simple_S)
#complex_vs_simple_30=stats.ttest_ind(complex_G,simple_S_30)
inclinations_0_vs_30=stats.ttest_ind(inclination_0,inclination_30)
#print(complex_vs_simple_0)
#print(complex_vs_simple_30)
#print(inclinations_0_vs_30)
#A=stats.ttest_1samp(G1_0, 0)
#B=stats.ttest_ind(G1_0,G1_15)
#C=stats.ttest_ind(G1_0,G1_30)
#D=stats.ttest_ind(G1_0,SC_0)
#E=stats.ttest_ind(G1_0,SC_30)
#F=stats.ttest_ind(G1_0,SS_0)
#G=stats.ttest_ind(G1_0,SS_30)
#H=stats.ttest_ind(G1_30,SC_30)
#I=stats.ttest_ind(G1_30,SS_30)
#J=stats.ttest_ind(G1_15,SC_30)
#stats.ttest_1samp(G1_0, popmean=0)
#P=stats.ttest_ind(G1_0,G1_0)

'''
# Define labels, positions, bar heights and error bar heights
#labels = ['G1_0', 'G1_15', 'G1_30', 'SC_0','SC_30', 'SS_0' ,'SS_30']
labels= ['complex', 'simple(30)']
x_pos = np.arange(len(labels))
k=[143.87,	125.82,	105.25]
k=np.mean(k)
l=[143.87,	125.82,	105.25]
l=np.std(k)
m= [108.80,134.43]
m=np.mean(m)
n=[140.90,90.54]
n=np.std(n)
#CTEs = [20.80,  	19.58,	16.73,	10.94,	8.70,	10.29,	7.99]
#CTEs = [143.87,	125.82,	105.25,	140.90,	108.80,	134.43,	90.54]
CTEs =[k,m]

#error = [3.76,	3.65,	2.57,	4.54,	3.18,	5.27,	3.18]
#error = [6.73,	6.11,	7.63,	32.13,	34.21,	20.09,	19.99]
error=[l,n]
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('(angle )')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('angle profile (complex vs simple(30) across five participants')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_angle_complex vs simple(30).png')
plt.show()
'''
'''
#Mean_simple=(SC_0 + SC_30 + SS_0 + SS_30)/4
#Mean_complex= ( G1_0 +  G1_15 + G1_30)/3
#Mean_zero=(G1_0 + SC_0 +SS_0)/3
#Mean_30 =( G1_30 +SC_30 + SS_30)/3
#with np.load(G1_0 , G1_30 ) as data:
#    
sns.set(style="darkgrid")
#tip1 = sns.load_dataset("G1_0")
#tip2 = sns.load_dataset("G1_30")
#complex=[]
#complex.append(G1_0)
#complex.append(G1_30
dcomplex=pd.DataFrame()

mylist=[]
caselist=[None]*10#*24
caselist[0:5]=['simple','simple','simple','simple','simple']#,'simple','simple','simple','simple','simple','simple','simple']
#caselist[0:5]=['inclination_0','inclination_0','inclination_0','inclination_0','inclination_0']#,'inclination_0','inclination_0','inclination_0','inclination_0','inclination_0','inclination_0','inclination_0']
caselist[5:10]=['complex','complex','complex','complex','complex']#,'complex','complex','complex','complex','complex','complex','complex']
#caselist[5:10]=['inclination_30','inclination_30','inclination_30','inclination_30','inclination_30']#,'inclination_30','inclination_30','inclination_30','inclination_30','inclination_30','inclination_30','inclination_30']
#mylist.extend(G1_0[0:,0])
mylist.extend(simple_s[0:,0])
#mylist.extend(SS_30[0:,0])
mylist.extend(complex_g[0:,0])
dcomplex['angle']= mylist
dcomplex['Task']= caselist
##dcomplex['simple']=SS_30[0:,0]#[5,6,7,8]
#tips = sns.load_dataset("tips")
ax = sns.pointplot(x="Task", y="angle",data=dcomplex, color="#bb3f3f")
#ax.set_title=('hagsdjhdh')
#ax = sns.pointplot("time", y="tip", data=tips,
#                   color="#bb3f3f")
'''




















