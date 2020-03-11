# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:59:03 2020

@author: USER
"""
import math
import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as py
class Kine3:
#position_data=pd.read_csv("Angle_test.csv")
#with open('Angle_test.csv') as f:
#    reader = csv.reader(f)
#    reader.next()
#    for row in reader:
#        for (i,v) in enumerate(row):
#            columns[i].append(v)
#print(columns[0])
f =open("Angle_test.txt", "r")
print(f.read(3))

position_data=pd.read_csv("Angle_test .csv")
a=position_data(:,12)
b=position_data(:,15)
c=position_data(:,18)  
def cosCalc(self, a, b, c):
angle = math.acos((a*a + b*b - c*c)/(2*a*b))
return angle
def inputCalc(self):
xe = float(input("xe: "))
ye = float(input("ye: "))
phie = math.radians(float(input("phie in degrees: ")))
l1 = float(input("l1: "))
l2 = float(input("l2: "))
l3 = float(input("l3: "))
xw = xe - l3 * math.cos(phie)
yw = ye - l3 * math.sin(phie)
r = math.sqrt(xw**2 + yw**2)
gamma = self.cosCalc(r, l1, l2)
theta2 = math.pi - self.cosCalc(l1, l2, r)
theta1 = math.atan2(yw, xw) - gamma
theta3 = phie - theta1 - theta2
print("theta1: {} and {}".format(math.degrees(theta1), math.degrees(theta1 + 2 * gamma)))
print("theta2: {} and {}".format(math.degrees(theta2), math.degrees(theta2 * -1)))
print("theta3: {} and {}".format(math.degrees(theta3), math.degrees(theta3 + 2 * (theta2 - gamma))))
kine3 = Kine3()
kine3.inputCalc()