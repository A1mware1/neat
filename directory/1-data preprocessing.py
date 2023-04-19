import sys
filesNames_CubeControl=[]
os.chdir("/content/drive/MyDrive/CubeAndSpiralData/NeckerCubeDrawings/patients")

for file in glob.glob("*.txt"):
    filesNames_CubeControl.append(file)

list = []

for name in filesNames_CubeControl:
  f=open(name)
  line = f.readline()
  while line:
    a = line.split("	")          
    b = a[1:3]              
    list.append(b)
    list.append('\n')
    line = f.readline()
  f.close()
  
  with open("/content/drive/MyDrive/CubeAndSpiralData1/NeckerCubeDrawings/patients/"+name, 'a') as month_file:
     
     for line in list:
         s = ' '.join(line)
         month_file.write(s)
  list=[]
