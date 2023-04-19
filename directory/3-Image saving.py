
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

#to make sequre images
matplotlib.rcParams['figure.figsize'] = (4, 4)



#create list of new files name 
import glob, os
filesNames_NewCubeControl=[]
os.chdir("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control")
for file in glob.glob("*.txt"):
    filesNames_NewCubeControl.append(file)



#link of the images folder
#draw and save the CubeControl images
dataPath_CubeControl='/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control/Images'
for name in filesNames_NewCubeControl:
    x = pd.read_csv(    name, sep=',',usecols=[0]    )
    y = pd.read_csv(    name, sep=',',usecols=[1]    )
    plt.scatter(x, y)
    plt.axis('off')
    plt.savefig('{}.png'.format(name))
    plt.close()



filesNames_CubePatients=[]
os.chdir('/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient')
for file in glob.glob("*.txt"):
    filesNames_CubePatients.append(file)


#draw and save the CubePatients images
for name in filesNames_CubePatients:
    x = pd.read_csv(    name, sep=',',usecols=[0]    )
    y = pd.read_csv(    name, sep=',',usecols=[1]    )
    plt.scatter(x, y)
    plt.axis('off')
    plt.savefig('{}.png'.format(name))
    plt.close()



#

