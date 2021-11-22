# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelBuild.py

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
ModelFolder="D:\CowBhave\Models"

TrainingDataFolder,TrainingDataSetName="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled","Barn2"
# TrainingDataFolder,TrainingDataSetName="D:\\CowBhave\\Data_Exp08_11_2019\\Labeled","Barn1"
# TrainingDataFolder,TrainingDataSetName="D:\\CowBhave\\Pavlovic21\\Labeled","Barn3"

Fold_i,Vers_i=0,0
ModelName="NN1"#"SVM1"#"RF1"#

f = h5py.File(TrainingDataFolder+"\\"+'FeedingBehaviour'+"WS"+str(WindowSize)+"F"+str(Freq)+'.h5', 'r')
ASlicedTraining = f['AccXYZ'][...]
LabelSlicedTraining = f['LabelOneHot'][...]
LabelSlicedValidation = f['Label'][...]
f.close()
print(ASlicedTraining.shape)
print(LabelSlicedTraining.shape)
print(ASlicedTraining)
print(LabelSlicedTraining)
print(LabelSlicedTraining)
print(numpy.unique(LabelSlicedValidation))

if ModelName in ["NN1", "NN2", "NN3", "NN4"]:
    print(ModelName+' '+'Training')
    epochs, batch_size = 20, 32

    model=CNNModelDefine(ModelName,ASlicedTraining,LabelSlicedTraining)
    model.fit(ASlicedTraining, LabelSlicedTraining, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(ModelFolder+"\\"+ModelName+TrainingDataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))

elif ModelName=="RF1" or ModelName=="SVM1":
    print(ModelName+' '+'Training')
    FeaturesTrain,LabelSlicedTraining=AccelerationFeatures(AxSlicedTraining,AySlicedTraining,AzSlicedTraining,LabelSlicedTraining)

    trainX, testX, trainy, testy=train_test_split(FeaturesTrain,LabelSlicedTraining,test_size=0.2, random_state=0, stratify=LabelSlicedTraining)
    print("Test sample number " + str(len(testX)) + ", train sample number " + str(len(trainX)))
    trainy=trainy - 1
    testy = testy - 1

    if ModelName=="RF1":
        clf=RandomForestClassifier(n_estimators=50) #n_estimators=250, Kaler 19; n_estimators=2000, ?max_features=15, Riabof 20; 8, 128, 1 Walton 19;
    elif ModelName=="SVM1":
        # clf=svm.SVC()
        # clf=svm.SVC(kernel='poly', degree=8)
        clf=svm.SVC(kernel='rbf', C=128, gamma=0.05) #Kaler 19; C=128, gamma=0.05, Riaboff 20; Trung 18;
        # clf=svm.SVC(kernel='sigmoid')

    clf.fit(trainX,trainy)
    joblib.dump(clf, ModelFolder+"\\"+ModelName+TrainingDataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i)+".joblib")
    # clf = joblib.load("./random_forest.joblib")

    print('Validating')
    FeaturesTest,LabelSlicedValidation=AccelerationFeatures(AxSlicedValidation,AySlicedValidation,AzSlicedValidation,LabelSlicedValidation)

    FeedingBehaviorPredicted=clf.predict(FeaturesTest)+1    

print(model.summary())
