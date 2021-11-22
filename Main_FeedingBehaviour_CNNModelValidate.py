# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelValidate.py

from FeedingBehavior_NNlib import *
import fnmatch, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import scipy.signal
import h5py

WindowSize=10 #sec
Freq=25 #Hz
WindowN=int(WindowSize*Freq)
print("Interval length " + str(WindowSize) + "sec")
FeedingM, RuminatingM, NothingM, DrinkingM = 1, 2, 3, 4

# TrainingDataSetName="Barn1"
TrainingDataSetName="Barn2"
# TrainingDataSetName="Barn3"
ValidationDataFolder,ValidationDataSetName="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled","Barn2"
# ValidationDataFolder,ValidationDataSetName="D:\\CowBhave\\Data_Exp08_11_2019\\Labeled","Barn1"
# ValidationDataFolder,ValidationDataSetName="D:\\CowBhave\\Pavlovic21\\Labeled","Barn3"

Fold_i,Vers_i=0,0
ModelName="NN1"#"SVM1"#"RF1"#
ModelFolder="D:\CowBhave\Models"
model=tensorflow.keras.models.load_model(ModelFolder+"\\"+ModelName+TrainingDataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))

f = h5py.File(ValidationDataFolder+"\\"+'FeedingBehaviour'+"WS"+str(WindowSize)+"F"+str(Freq)+'.h5', 'r')
ASlicedValidation = f['AccXYZ'][...]
LabelSlicedValidation = f['Label'][...]
f.close()
ASlicedValidation = ASlicedValidation.reshape((len(ASlicedValidation), WindowN, 3))
LabelSlicedValidation  = LabelSlicedValidation.reshape((len(LabelSlicedValidation), 1))

print(model.summary())
# print(ASlicedValidation.shape)
# print(LabelSlicedValidation.shape)
print(numpy.unique(LabelSlicedValidation))

print('Validating')
FeedingBehaviorPredictedP=model.predict(ASlicedValidation)
FeedingBehaviorPredicted=list()
for i in range(len(FeedingBehaviorPredictedP)):
    m,k=MaxInd(FeedingBehaviorPredictedP[i])
    FeedingBehaviorPredicted.append(k+1)

FeedingBehaviorPredicted=scipy.signal.medfilt(FeedingBehaviorPredicted,5)
PerfVect, ConfusionMatr, TotAcc=PresentPerformance(LabelSlicedValidation,FeedingBehaviorPredicted,[FeedingM, RuminatingM, NothingM],["Feeding", "Ruminating", "Nothing"])
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")
s=dt_string+";"+TrainingDataSetName+";"+ValidationDataSetName+";"+str(WindowSize)+";"+str(Fold_i)+";"+str(Vers_i)+";"+str(TotAcc)+";"
for i in range(len(ConfusionMatr)):
    for j in range(len(ConfusionMatr)):
        s=s+str(ConfusionMatr[i,j])+";"
fileH=open(ModelFolder+"\\"+ModelName+'_Performance.csv', 'a')
fileH.write(s+'\n')
fileH.close()

print('End')