from PIL import Image
from numpy import array
import  numpy
from keras.utils import np_utils
from skimage import io
import cv2
import glob, os
filesNames_CubeControl=[]
# images
os.chdir("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control/32x32")

for file in glob.glob("*.png"):
    filesNames_CubeControl.append(file)
# RGBA images
input_CNN=[]
output_CNN=[]

for imageName in filesNames_CubeControl:
    temp=[]
    img = cv2.imread(imageName,0)
    temp.append(img)
    arr=array(temp)
    input_CNN.append(arr)
    output_CNN.append([0])

#array(dataInput_CubeControl)
print (len(input_CNN))
print (len(output_CNN))

# create list of  files name 
import glob, os
filesNames_CubePatient=[]
os.chdir("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient/32x32")

for file in glob.glob("*.png"):
    filesNames_CubePatient.append(file)

#appent patients fata to the control one
#read B&W 32x32 images 
for imageName2 in filesNames_CubePatient:
    temp2=[]
    img2 = cv2.imread(imageName2,0)
    temp2.append(img2)
    arr2=array(temp2)
    input_CNN.append(arr2)
    output_CNN.append([1])
    
#array(dataInput_CubeControl)
print (len(input_CNN))
print (len(output_CNN))

from tempfile import TemporaryFile
outfile = TemporaryFile()
numpy.save("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/input_CNN_cube.npy", input_CNN)
numpy.save("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/output_CNN_cube.npy", output_CNN)