# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:19:10 2020

@author: asus
"""



import os
import pandas as pd



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
    
       

for each_map in mappinglist:
    



"""
subj1epchd=[]
subj2epchd=[]
subj3epchd=[]
subj4epchd=[]
subj5epchd=[]
subj6epchd=[]
subj7epchd=[]
subj8epchd=[]
subj9epchd=[]
subj10epchd=[]
subj11epchd=[]
subj12epchd=[]



for file in subjectfile:
    
    idxs=(file[41:43])
    #print(idxs)
    if (idxs=='04'):
       subj1epchd.append(file)    
    if (idxs=='05'):
       subj2epchd.append(file)
    if (idxs=='06'):
       subj3epchd.append(file)
    if (idxs=='07'):
       subj4epchd.append(file)
    if (idxs=='08'):
       subj5epchd.append(file) 
    if (idxs=='10'):
       subj6epchd.append(file)
    if (idxs=='12'):
       subj7epchd.append(file) 
    if (idxs=='13'):
       subj8epchd.append(file) 
    if (idxs=='14'):
       subj9epchd.append(file)  
    if (idxs=='15'):
       subj10epchd.append(file)
    if (idxs=='16'):
       subj11epchd.append(file)
    if (idxs=='17'):
       subj12epchd.append(file)

cond_subfiles=[subj1epchd, subj2epchd,subj3epchd,subj4epchd,subj5epchd,subj6epchd,
               subj7epchd,subj8epchd,subj9epchd,subj10epchd,subj11epchd,subj12epchd ]


subj1epchd=[]
subj2epchd=[]
subj3epchd=[]
subj4epchd=[]
subj5epchd=[]
subj6epchd=[]
subj7epchd=[]
subj8epchd=[]
subj9epchd=[]
subj10epchd=[]
subj11epchd=[]
subj12epchd=[]


for file in epochfile:
    idxs=(file[31:33])
   
    if (idxs=='1_'):
       subj1epchd.append(file)
    if ((idxs[0])==str(2)):
        subj2epchd.append(file)
    if (idxs[0]) ==str(3) :
        subj3epchd.append(file)
    if (idxs[0]) ==str(4) :
        subj4epchd.append(file)
    if (idxs[0]) ==str(5) :
        subj5epchd.append(file)
    if (idxs[0]) ==str(6) :
        subj6epchd.append(file)    
    if (idxs[0]) ==str(7) :
        subj7epchd.append(file)    
    if (idxs[0]) ==str(8) :
        subj8epchd.append(file)    
    if (idxs[0]) ==str(9) :
        subj9epchd.append(file)
    if (idxs[0:2]) ==str(10)  :
        subj10epchd.append(file)
    if (idxs[0:2]) ==str(11)  :
        subj11epchd.append(file)
    if (idxs[0:2]) ==str(12)  :
        subj12epchd.append(file)


cond_subfiles_epochs=[subj1epchd, subj2epchd,subj3epchd,subj4epchd,subj5epchd,subj6epchd,
               subj7epchd,subj8epchd,subj9epchd,subj10epchd,subj11epchd,subj12epchd ]

#
#
#    
#cond_epoched=pd.DataFrame()
#i=0   
#for subject in subjectfile:
#    
##    if i
#    raw_data=subject
#    idx=raw_data[41:43]
#    print(idx ,idxmap[idx])
#    
#    epoch_data=epochfiles.iloc[:,idxmap[idx]-1]
#    cond_epoched[ epochfiles.keys()[i]+"_circle_0"]=epoch_data[0:8]
#    cond_epoched[ epochfiles.keys()[i]+"_circle_30"]=epoch_data[8:16]
#    cond_epoched[ epochfiles.keys()[i]+"_randomgame_0"]=epoch_data[16]
#    cond_epoched[ epochfiles.keys()[i]+"_randomgame_15"]=epoch_data[17]
#    cond_epoched[ epochfiles.keys()[i]+"_randomgame_30"]=epoch_data[18]
#    cond_epoched[ epochfiles.keys()[i]+"_star_0"]=epoch_data[19:27]
#    cond_epoched[ epochfiles.keys()[i]+"_star_30"]=epoch_data[27:35]
#    i=i+1
#    
##    print(raw_data)
##    print(epoch_data)
##    
##    print('\n')

##
#for each_subj in cond_subfiles:
#    data_files=each_subj
#    for each_file in data_files:
#        position_data= pd.read_csv((each_file), sep="\t",skiprows=8)
#        for each_epochs in cond_subfiles_epochs:
#            epoch_list=each_epochs
#            for each_epoch in epoch_list:
#                print(each_file[41:50],each_epoch[31:46])
#                print('\n')
#        

"""