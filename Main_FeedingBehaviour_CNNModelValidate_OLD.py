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

WindowSize=10 #sec
Freq=25 #Hz
WindowN=int(WindowSize*Freq)
print("Interval length " + str(WindowSize) + "sec")
FeedingM, RuminatingM, NothingM, DrinkingM = 1, 2, 3, 4

# TrainingDataFolder="D:\CowBhave\Data_Exp08_11_2019\Labeled"
TrainingDataFolder,TrainingDataSetName="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled","Barn2"
# TrainingDataFolder="D:\\CowBhave\\Pavlovic21\\Labeled"
ValidationDataFolder,ValidationDataSetName="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled","Barn2"
# ValidationDataFolder,ValidationDataSetName="D:\\CowBhave\\Data_Exp08_11_2019\\Labeled","Barn1"
# ValidationDataFolder,ValidationDataSetName="D:\\CowBhave\\Pavlovic21\\Labeled","Barn3"
# ValidationDataFolder,ValidationDataSetName="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled","Barn2"

Fold_i,Vers_i=0,0
ModelName="NN1"#"SVM1"#"RF1"#
ModelFolder="D:\CowBhave\Models"
model=tensorflow.keras.models.load_model(ModelFolder+"\\"+ModelName+TrainingDataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))
print('Reading')
TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label = ReadLabeledDataFiles(ValidationDataFolder,'AccDataLabeled_Tag',Freq)
AxSlicedValidation, AySlicedValidation, AzSlicedValidation, LabelSlicedValidation, TagNoSliced, CowNoSliced, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
print('Validating')
ASlicedValidation=numpy.dstack((AxSlicedValidation,AySlicedValidation,AzSlicedValidation))
FeedingBehaviorPredictedP=model.predict(ASlicedValidation)
FeedingBehaviorPredicted=list()
for i in range(len(FeedingBehaviorPredictedP)):
    m,k=MaxInd(FeedingBehaviorPredictedP[i])
    FeedingBehaviorPredicted.append(k+1)

# print('Reading')
# TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label = ReadLabeledDataFiles(TrainingDataFolder,'AccDataLabeled_Tag',Freq)
# CowNoList=numpy.unique(CowNo)
# print('Cow no list')
# print(CowNoList)
# print('Slicing')
# AxSlicedTraining, AySlicedTraining, AzSlicedTraining, LabelSlicedTraining, TagNoSliced, CowNoSlicedTraining, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
# print('Balancing')
# AxSlicedTraining, AySlicedTraining, AzSlicedTraining, LabelSlicedTraining, CowNoSlicedTraining=DataBalance(AxSlicedTraining, AySlicedTraining, AzSlicedTraining, LabelSlicedTraining, CowNoSlicedTraining)
# AxSlicedTraining, AySlicedTraining, AzSlicedTraining, LabelSlicedTraining, CowNoSlicedTraining=AccAugmentation(AxSlicedTraining, AySlicedTraining, AzSlicedTraining, LabelSlicedTraining, CowNoSlicedTraining, 5)


# if ModelName in ["NN1", "NN2", "NN3", "NN4"]:
#     print(ModelName+' '+'Training')
#     epochs, batch_size = 20, 32

#     ASlicedTraining=numpy.dstack((AxSlicedTraining,AySlicedTraining,AzSlicedTraining))
#     LabelSlicedTraining =numpy.asarray(LabelSlicedTraining) - 1
#     LabelSlicedTraining = to_categorical(LabelSlicedTraining)

#     model=CNNModelDefine(ModelName,ASlicedTraining,LabelSlicedTraining)
#     model.fit(ASlicedTraining, LabelSlicedTraining, epochs=epochs, batch_size=batch_size, verbose=0)
#     model.save(ModelFolder+"\\"+ModelName+TrainingDataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))

#     print('Validating')
#     ASlicedValidation=numpy.dstack((AxSlicedValidation,AySlicedValidation,AzSlicedValidation))
#     FeedingBehaviorPredictedP=model.predict(ASlicedValidation)
#     FeedingBehaviorPredicted=list()
#     for i in range(len(FeedingBehaviorPredictedP)):
#         m,k=MaxInd(FeedingBehaviorPredictedP[i])
#         FeedingBehaviorPredicted.append(k+1)

# elif ModelName=="RF1" or ModelName=="SVM1":
#     print(ModelName+' '+'Training')
#     FeaturesTrain,LabelSlicedTraining=AccelerationFeatures(AxSlicedTraining,AySlicedTraining,AzSlicedTraining,LabelSlicedTraining)

#     trainX, testX, trainy, testy=train_test_split(FeaturesTrain,LabelSlicedTraining,test_size=0.2, random_state=0, stratify=LabelSlicedTraining)
#     print("Test sample number " + str(len(testX)) + ", train sample number " + str(len(trainX)))
#     trainy=trainy - 1
#     testy = testy - 1

#     if ModelName=="RF1":
#         clf=RandomForestClassifier(n_estimators=50) #n_estimators=250, Kaler 19; n_estimators=2000, ?max_features=15, Riabof 20; 8, 128, 1 Walton 19;
#     elif ModelName=="SVM1":
#         # clf=svm.SVC()
#         # clf=svm.SVC(kernel='poly', degree=8)
#         clf=svm.SVC(kernel='rbf', C=128, gamma=0.05) #Kaler 19; C=128, gamma=0.05, Riaboff 20; Trung 18;
#         # clf=svm.SVC(kernel='sigmoid')

#     clf.fit(trainX,trainy)
#     joblib.dump(clf, ModelFolder+"\\"+ModelName+TrainingDataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i)+".joblib")
#     # clf = joblib.load("./random_forest.joblib")

#     print('Validating')
#     FeaturesTest,LabelSlicedValidation=AccelerationFeatures(AxSlicedValidation,AySlicedValidation,AzSlicedValidation,LabelSlicedValidation)

#     FeedingBehaviorPredicted=clf.predict(FeaturesTest)+1    

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


# print('End')