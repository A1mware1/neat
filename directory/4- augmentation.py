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
from keras.utils import image_utils, load_img,img_to_array
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')



# create list of  files name 
import glob, os
filesNames_CubeControl=[]

os.chdir("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control/32x32")

for file in glob.glob("*.png"):
    filesNames_CubeControl.append(file)


#augmentation process for control
path="/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control/Aug"
for imageName in filesNames_CubeControl:
    img = load_img(imageName)  # this is a PIL image
    x = img_to_array(img) 
    x = x.reshape((1,) + x.shape)
    i = 0
    datagen.flow(x, batch_size=1,save_to_dir=path, save_prefix='Aug', save_format='png')
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=path, save_prefix='Aug', save_format='png'):
        i += 1
        if i > 22:
            break  # otherwise the generator would loop indefinitely


# create list of  files name 
import glob, os
filesNames_CubePatient=[]
os.chdir("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient/32x32")

for file2 in glob.glob("*.png"):
    filesNames_CubePatient.append(file2)



#augmentation process for paitents

path2="/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient/Aug"
for imageName2 in filesNames_CubePatient:
    img2 = load_img(imageName2)  # this is a PIL image
    x2 = img_to_array(img2) 
    x2 = x2.reshape((1,) + x2.shape)
    i2 = 0
    datagen.flow(x2, batch_size=1,save_to_dir=path2, save_prefix='Aug', save_format='png')
    for batch2 in datagen.flow(x2, batch_size=1,
                          save_to_dir=path2, save_prefix='Aug', save_format='png'):
        i2 += 1
        if i2 > 10:
            break  # otherwise the generator would loop indefinitely

