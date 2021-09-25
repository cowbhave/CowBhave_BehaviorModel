# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelPredictAndCompare.py
from FeedingBehavior_NNlib import *
import fnmatch, os
from sklearn.model_selection import train_test_split
import numpy
import scipy.signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import tensorflow
from datetime import datetime
# import matplotlib.pyplot as plt

FoldCowNoList=[[649, 683, 686, 715],
               [414, 683, 686, 715],
               [414, 649, 686, 715],
               [414, 649, 683, 715],
               [414, 649, 683, 686]]
FoldCowNoList=[[414],
               [649],
               [683],
               [686],
               [715]]

WindowSize=10 #sec
Freq=25 #Hz
WindowN=int(WindowSize*Freq)
print("Interval length " + str(WindowSize) + "sec")
FeedingM, RuminatingM, NothingM, DrinkingM = 1, 2, 3, 4

DataFolder="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"
# DataFolder="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"
DataSetName="Barn2"
ModelFolder="D:\CowBhave\Models"

Fold_i=0
Vers_i=1
model=tensorflow.keras.models.load_model(ModelFolder+"\\"+DataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))

print('Reading')
TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label = ReadLabeledDataFiles(DataFolder,'AccDataLabeled_Tag')
# plt.scatter(range(len(Label)), Label)
# plt.show()

print('Slicing')
AxSliced, AySliced, AzSliced, LabelSliced, TagNoSliced, CowNoSliced, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 1, TagNo, CowNo, TimeStamp)
for Fold_i in range(len(FoldCowNoList[0])*0+1):
    print("Fold "+str(Fold_i)+", cows "+str(FoldCowNoList[Fold_i]))
    AxSlicedFold,AySlicedFold,AzSlicedFold,LabelSlicedFold=[],[],[],[]
    for i in range(len(LabelSliced)):        
        if CowNoSliced[i] in FoldCowNoList[Fold_i]:
            AxSlicedFold.append(AxSliced[i])
            AySlicedFold.append(AySliced[i])
            AzSlicedFold.append(AzSliced[i])
            LabelSlicedFold.append(LabelSliced[i])#
        ASlicedFold=numpy.dstack((AxSlicedFold,AySlicedFold,AzSlicedFold))
    
    print('Predicting')
    FeedingBehaviorPredictedP=model.predict(ASlicedFold)
    FeedingBehaviorPredicted=list()
    for i in range(len(FeedingBehaviorPredictedP)):
        m,k=MaxInd(FeedingBehaviorPredictedP[i])
        FeedingBehaviorPredicted.append(k+1)

    FeedingBehaviorPredicted=scipy.signal.medfilt(FeedingBehaviorPredicted,5)
    PerfVect, ConfusionMatr=PresentPerformance(LabelSlicedFold,FeedingBehaviorPredicted,[FeedingM, RuminatingM, NothingM],["Feeding", "Ruminating", "Nothing"])

    now = datetime.now()
    dt_string = now.strftime("%YYYY%mm%dd%HH%MM")
    s=dt_string+";"+DataSetName+";"+str(WindowSize)+";"+str(Fold_i)+";"+str(Vers_i)+";"
    for i in range(len(ConfusionMatr)):
        for j in range(len(ConfusionMatr)):
            s=s+str(ConfusionMatr[i,j])+";"
    fileH=open(ModelFolder+"\\"+'Model1_Performance.csv', 'a')
    fileH.write(s)
    fileH.close()


# LabelSliced=LabelSlicedFold
# ASliced=numpy.dstack((AxSliced,AySliced,AzSliced))

# plt.scatter(range(len(LabelSliced)), LabelSliced)
# plt.show()

# print('Predicting')
# FeedingBehaviorPredictedP=model.predict(ASliced)

# # FeedingBehaviorPredicted, max_value = max(FeedingBehaviorPredictedP)
# FeedingBehaviorPredicted=list()
# for i in range(len(FeedingBehaviorPredictedP)):
#     m,k=MaxInd(FeedingBehaviorPredictedP[i])
#     FeedingBehaviorPredicted.append(k+1)

# FeedingBehaviorPredicted=scipy.signal.medfilt(FeedingBehaviorPredicted,5)

# PerfVect, ConfusionMatr=PresentPerformance(LabelSliced,FeedingBehaviorPredicted,[FeedingM, RuminatingM, NothingM],["Feeding", "Ruminating", "Nothing"])

