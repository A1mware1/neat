!pip install python-resize-image
from PIL import Image

from resizeimage import resizeimage




# create list of  files name 
import glob, os
filesNames_CubeControl=[]
# os.chdir("/home/alanauser/Desktop/Parkinsion project/Dataset/Cube/Control/Images")
os.chdir("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control")

for file in glob.glob("*.png"):
    filesNames_CubeControl.append(file)



#reszie process for control

path="/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control/32x32"

for imagee in filesNames_CubeControl:
    with open(imagee, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_contain(image, [256,256])
            cover.save("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Control/32x32/ {}.png".format(imagee),image.format)



# create list of  files name 
import glob, os
filesNames_CubePatient=[]

os.chdir("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient")
for file in glob.glob("*.png"):
    filesNames_CubePatient.append(file)


#resize process for patients

path2="/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient/32x32"

for imagee in filesNames_CubePatient:
    with open(imagee, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_contain(image, [256, 256])
            cover.save("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/Patient/32x32/{}.png".format(imagee), image.format)

