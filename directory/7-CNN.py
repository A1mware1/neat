from keras.utils import image_utils, load_img,img_to_array
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.image_data_format() == 'channel_first'
from keras.constraints import maxnorm

from PIL import Image
from numpy import array
import  numpy
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD

import glob, os
import matplotlib.pyplot as plt

import random
from time import time

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#CNN input 
Input_CNN = numpy.load("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/input_CNN_cube.npy")
output_CNN = numpy.load("/content/drive/MyDrive/Parkinsion project/Dataset/RNN datasets/Cube/output_CNN_cube.npy")


from sklearn.model_selection import train_test_split
def split(data):
  x_train ,x_test = train_test_split(data,test_size=0.1) 
  return x_train ,x_test

#shuffle the input&output and divide to 90% training & valdation and 10% testing
data_xxx=Input_CNN
data_yyy=output_CNN

c = list(zip(data_xxx, data_yyy))
random.shuffle(c)
cc, newtest=split(c)
data_xx, data_yy = zip(*cc)
newtestX, newtestY = zip(*newtest)
data_x=array(data_xx)
data_y=array(data_yy)
newtestX=array(newtestX)
newtestY=array(newtestY)

data_x = data_x.astype('float32')
newtestX = newtestX.astype('float32')
data_x = data_x / 255.0
newtestX = newtestX / 255.0

#Images normaliztion
from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(featurewise_center=True, 
                               featurewise_std_normalization=True)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)



#SGD Optimizer
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)

#calc specificity function 

def specificity(TN, FP):
    return float(TN)/float(TN+FP)

#calc trainTime function  
  
def trainTime(t1, t2):
    t=float(t2)-float(t1)
    tMin=float(t/60)
    return tMin

#calc kappa function  

def kappa(TN, FP, FN, TP): 
  total=TP+ FP+ TN+ FN
  oA=(TN+TP)/total
  eA1=(TN+FP)*(TN+FN)
  eA2=(TP+FN)*(FP+TP)
  eA=(eA1+eA2)/(total*total)
  kappa=(oA-eA)/(1-eA)
  return kappa


#draw precsion_recall plot

def prec_rec(y_test, y_score):
  global cv
  precision, recall, _ = precision_recall_curve(y_test, y_score)
  plt.subplot(111)
  plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
  plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('2-class Precision-Recall curve: AP={0:0.2f}, For CV={1:10}'.format(
          average_precision,cv))
  plt.savefig('{}.png'.format('Precision-Recall CV '+ str(cv)))
  plt.close()


#draw train accuracy & error vs epoch _ plot val acc and error vs epoch function  

def acc_epo(history):
#    print(history.history.keys())
#    history_dict = history.history
#    history_dict.keys()

    global cv, cvscores
    cv=cv+1
#    os.chdir(results_path)
# summarize history for accuracy
    plt.subplot(111)
    plt.plot(history.history['accuracy'])
#    plt.plot(history.history['val_acc'])
    plt.axhline(cvscores[cv-1]/100, color="gray")
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
    plt.title('model train-accuracy and train-loss CV '+ str(cv))
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')

    plt2=plt.twinx()
    plt2.plot(history.history['loss'],'r-')    
    plt2.set_ylabel('Error', color='r')
    for tl in plt2.get_yticklabels():
      tl.set_color('r')
    plt.legend(['Train Error'], loc='upper right')
    plt.savefig('{}.png'.format('model train-accuracy and train-loss CV '+ str(cv)))
    plt.close()
    
    
    #plot val acc and error
    plt.subplot(111)
    plt.plot(history.history['val_accuracy'])    
    plt.axhline(cvscores[cv-1]/100, color="gray")
    plt.legend(['Valdation Accuracy', 'Test Accuracy'], loc='upper left')
    plt.title('model val-accuracy and val-loss CV '+ str(cv))
    plt.ylabel('Val-Accuracy')
    plt.xlabel('epoch')
    
    plt3=plt.twinx()
    plt3.plot(history.history['val_loss'],'r-')    
    plt3.set_ylabel('Val-Error', color='r')
    for tl in plt3.get_yticklabels():
      tl.set_color('r')
    plt.legend(['Valdation Error'], loc='upper right')
    plt.savefig('{}.png'.format('model val-accuracy and val-loss CV '+ str(cv)))
    plt.close()
  
    
# compite CNN
 
def cnnCompile(model):
 
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

#CNN train function      
def cnnFit(model,trainingx,train,testingx,test):  
    
    history = model.fit(trainingx, data_y[train],validation_data=(testingx, data_y[test]),batch_size= 16,epochs=150, verbose=0)
    
    return history


 #calc classification accuracy 
  
def cal_conf(model,newtestx,newtestY):
# this for sigmoid activiation function since it gives continous value so we need threshold
    Y_pred = np.squeeze(model.predict(newtestx))
    threshold = 0.5
    print(classification_report(newtestY, Y_pred > threshold))
 
    return Y_pred

def cnnModel2():	
  # Create the model
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(1,256, 256), activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(Dropout(0.2))
  model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))

  return model


#normaliztion process
def normalization(train,test):
  global newtestX,data_x
  
  trainingData = data_x
  valdaitonData = data_x
  testingData = newtestX
  
  # Calculate statistics on train dataset
  generator.fit(data_x[train])
  # Apply featurewise_center to test-data with statistics from train data
  trainingData[train] -= generator.mean
  # Apply featurewise_std_normalization to test-data with statistics from train data
  trainingData[train] /= (generator.std + K.epsilon())

  # Apply featurewise_center to test-data with statistics from train data
  valdaitonData[test] -= generator.mean
  # Apply featurewise_std_normalization to test-data with statistics from train data
  valdaitonData[test] /= (generator.std + K.epsilon())
  
  # Apply featurewise_center to test-data with statistics from train data
  testingData -= generator.mean
  # Apply featurewise_std_normalization to test-data with statistics from train data
  testingData /= (generator.std + K.epsilon())
  return trainingData[train],valdaitonData[test], testingData

split=kfold.split(data_x, data_y)

#cross-valdation process
cvscores = []
#to know which cross-val we are now to plot the acc_epoch
cv=0

for train, test in split:
     
    #image normalizion
    trainingx, testingx,newtestx=normalization(train,test)
    
    #create the model
    model=cnnModel2()
    
    #compile CNN
    cnnCompile(model)
    
    #CNN fit
    t1=time()   
    history=cnnFit(model,trainingx, train,testingx,test)
    t2=time()
    
    #calc accuracy 
    scores =model.evaluate(newtestx, newtestY, verbose=0)
    print('--------------------------------------------------')
    print("CrossValdation - Fold No# ", cv+1)
    print("Classification Test","%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)  
    print("Classification Test","%s: %.2f%%" % ('Error', scores[0]*100))
    #I can add here the loosscore=[] then append scores[0]*100 to it and plot it as horizintal line

    
    
    #calc confusion matrix
    Y_pred=cal_conf(model,newtestx,newtestY)

    #Draw accuracy & Error vs Epoches
    acc_epo(history)  

    # Calc AP
    average_precision = average_precision_score(newtestY, Y_pred )
    # Draw Precion vs Recall
    prec_rec(newtestY, Y_pred )    
    

    #Calc Mertics
    #this for sigmoid acticvaton function
    TN, FP, FN, TP= confusion_matrix(newtestY, Y_pred>0.5).ravel() 
    #this for sofmax acitivation function
    #Calc specificity and kappa
    print("%s: %.2f%%" % ('specificity', specificity(TN,FP)*100))
    print("%s: %.2f%%" % ('kappa', kappa(TN, FP, FN, TP)*100)) 
    
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

    print("The Confusion Matrix is: ",'TP=',TP,'FP=', FP,'TN=', TN,'FN=', FN)


    print("Trining time in minutes is: ",trainTime(t1, t2))
 
    
print('\n',"The over all is ")   
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))    
