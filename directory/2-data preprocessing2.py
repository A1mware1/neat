import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import requests
import io
from sklearn.metrics import mean_squared_error
import operator
import csv
import datetime
import glob, os
import math


#Cube dataset

#create list of files name (orginals files)
filesNames_CubeControl=[]
os.chdir("/content/drive/MyDrive/CubeAndSpiralData/NeckerCubeDrawings/controls")

for file in glob.glob("*.txt"):
    filesNames_CubeControl.append(file)

# check NULL value function
def checkNAN(value):
    newValue=value
    newValue = float(newValue)
    if math.isnan(newValue):
        value =0
    return value




#link of the procesesed dataset folder 
dataPath_CubeControl='/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control'
#read file name from the list we create in previous step 
for name in filesNames_CubeControl:
 with open(name) as data:
    oldFile =  csv.reader(data, delimiter='\t')
    row_count = sum(1 for row in oldFile)
    if row_count !=0:
        dd=dataPath_CubeControl,name
        newFilePath=os.path.join(*dd)
        myfile=open(newFilePath, "w")
        newFile = csv.writer(myfile)
        newFile.writerow(["timeStamp", "X", "Y","angle x","angle y","pressure"])
        with open(name) as data:
            oldFilee =  csv.reader(data, delimiter='\t')
            for elem in oldFilee:
# uncomment to exclude the zero-presure information                
              if(elem[5]!="0.0"):
               newFile.writerow([checkNAN(elem[0]),
               checkNAN(elem[1]),
                checkNAN(elem[2]),
                checkNAN(elem[3]),
                checkNAN(elem[4]),
                checkNAN(elem[5])])
               myfile.flush()
            myfile.close()



filesNames_CubePatients=[]
os.chdir("/content/drive/MyDrive/CubeAndSpiralData/NeckerCubeDrawings/controls")
for file in glob.glob("*.txt"):
    filesNames_CubePatients.append(file)

#link of the procesesed dataset folder 
dataPath_CubePatients='/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient'

for name in filesNames_CubePatients:
 with open(name) as data:
    d =  csv.reader(data, delimiter='\t')
    row_count = sum(1 for row in d)
    if row_count !=0:
        dd=dataPath_CubePatients,name
        pa=os.path.join(*dd)
        myfile=open(pa, "w")
        f = csv.writer(myfile)
        f.writerow(["timeStamp", "X", "Y","angle x","angle y","pressure"])
        with open(name) as data:
            ddd =  csv.reader(data, delimiter='\t')
            for elem in ddd:
# uncomment to exclude the zero-presure information  
              if(elem[5]!="0.0"):
               f.writerow([checkNAN(elem[0]),
               checkNAN(elem[1]),
                checkNAN(elem[2]),
                checkNAN(elem[3]),
                checkNAN(elem[4]),
                checkNAN(elem[5])])
               myfile.flush()
            myfile.close()



#Pentagon dataset



