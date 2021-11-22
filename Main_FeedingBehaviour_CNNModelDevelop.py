# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelDevelop.py

from FeedingBehavior_NNlib import *
import fnmatch, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
# from tensorflow.keras.utils import to_categorical
# import tensorflow
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import scipy.signal
from sklearn.model_selection import KFold

WindowSize=10 #sec
Freq=25 #Hz
WindowN=int(WindowSize*Freq)
print("Interval length " + str(WindowSize) + "sec")
FeedingM, RuminatingM, NothingM, DrinkingM = 1, 2, 3, 4

# TrainingDataFolder="D:\CowBhave\Data_Exp08_11_2019\Labeled"
TrainingDataFolder,TrainingDataSetName="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled","Barn2 interpolated frequency"
# TrainingDataFolder="D:\\CowBhave\\Pavlovic21\\Labeled"

ModelName="NN1"#"SVM1"#"RF1"#
ModelFolder="D:\CowBhave\Models"
print('Reading')
TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label = ReadLabeledDataFiles(TrainingDataFolder,'AccDataLabeled_Tag',Freq)
CowNoList=numpy.unique(CowNo)
print('Cow no list')
print(CowNoList)
print('Slicing')
AxSliced, AySliced, AzSliced, LabelSliced, TagNoSliced, CowNoSliced, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)

kf = KFold(n_splits=10)
Fold_i=0
for train_index, test_index in kf.split(CowNoList):
    FoldCowNoList=CowNoList[train_index]
    Fold_i=Fold_i+1
    Vers_i=0
    print("Cows "+str(FoldCowNoList))
    print("Fold "+str(Fold_i)+", version "+str(Vers_i))

    AxSlicedTrainingFold,AySlicedTrainingFold,AzSlicedTrainingFold,LabelSlicedTrainingFold,CowNoSlicedTrainingFold=[],[],[],[],[]
    AxSlicedValidationFold,AySlicedValidationFold,AzSlicedValidationFold,LabelSlicedValidationFold=[],[],[],[]
    for i in range(len(LabelSliced)):        
        if CowNoSliced[i] in FoldCowNoList:
            AxSlicedTrainingFold.append(AxSliced[i])
            AySlicedTrainingFold.append(AySliced[i])
            AzSlicedTrainingFold.append(AzSliced[i])
            LabelSlicedTrainingFold.append(LabelSliced[i])
            CowNoSlicedTrainingFold.append(CowNoSliced[i])
        else:
            AxSlicedValidationFold.append(AxSliced[i])
            AySlicedValidationFold.append(AySliced[i])
            AzSlicedValidationFold.append(AzSliced[i])
            LabelSlicedValidationFold.append(LabelSliced[i])
        
    print('Balancing')
    AxSlicedTrainingFold, AySlicedTrainingFold, AzSlicedTrainingFold, LabelSlicedTrainingFold, CowNoSlicedTrainingFold=DataBalance(AxSlicedTrainingFold, AySlicedTrainingFold, AzSlicedTrainingFold, LabelSlicedTrainingFold, CowNoSlicedTrainingFold)
    AxSlicedTrainingFold, AySlicedTrainingFold, AzSlicedTrainingFold, LabelSlicedTrainingFold, CowNoSlicedTrainingFold=AccAugmentation(AxSlicedTrainingFold, AySlicedTrainingFold, AzSlicedTrainingFold, LabelSlicedTrainingFold, CowNoSlicedTrainingFold, 5)
    
    if ModelName in ["NN1", "NN2", "NN3", "NN4"]:
        print(ModelName+' '+'Training')
        epochs, batch_size = 20, 32

        ASlicedTrainingFold=numpy.dstack((AxSlicedTrainingFold,AySlicedTrainingFold,AzSlicedTrainingFold))
        # trainX, testX, trainy, testy=train_test_split(ASlicedTrainingFold,LabelSlicedTrainingFold,test_size=0.2, random_state=0, stratify=LabelSlicedTrainingFold)
        # print("Test sample number " + str(len(testX)) + ", train sample number " + str(len(trainX)))
        LabelSlicedTrainingFold =numpy.asarray(LabelSlicedTrainingFold) - 1
        # testy = numpy.asarray(testy) - 1
        LabelSlicedTrainingFold = to_categorical(LabelSlicedTrainingFold)
        # testy = to_categorical(testy)

        model=CNNModelDefine(ModelName,ASlicedTrainingFold,LabelSlicedTrainingFold)
        model.fit(ASlicedTrainingFold, LabelSlicedTrainingFold, epochs=epochs, batch_size=batch_size, verbose=0)
        # _, score = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        # print('>#%d: %.3f' % (Fold_i+1, score* 100.0))
        model.save(ModelFolder+"\\"+ModelName+TrainingDataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))

        print('Validating')
        ASlicedValidationFold=numpy.dstack((AxSlicedValidationFold,AySlicedValidationFold,AzSlicedValidationFold))
        FeedingBehaviorPredictedP=model.predict(ASlicedValidationFold)
        FeedingBehaviorPredicted=list()
        for i in range(len(FeedingBehaviorPredictedP)):
            m,k=MaxInd(FeedingBehaviorPredictedP[i])
            FeedingBehaviorPredicted.append(k+1)

    elif ModelName=="RF1" or ModelName=="SVM1":
        print(ModelName+' '+'Training')
        FeaturesTrainFold,LabelSlicedTrainingFold=AccelerationFeatures(AxSlicedTrainingFold,AySlicedTrainingFold,AzSlicedTrainingFold,LabelSlicedTrainingFold)

        trainX, testX, trainy, testy=train_test_split(FeaturesTrainFold,LabelSlicedTrainingFold,test_size=0.2, random_state=0, stratify=LabelSlicedTrainingFold)
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
        FeaturesTestFold,LabelSlicedValidationFold=AccelerationFeatures(AxSlicedValidationFold,AySlicedValidationFold,AzSlicedValidationFold,LabelSlicedValidationFold)

        FeedingBehaviorPredicted=clf.predict(FeaturesTestFold)+1    

    FeedingBehaviorPredicted=scipy.signal.medfilt(FeedingBehaviorPredicted,5)
    PerfVect, ConfusionMatr, TotAcc=PresentPerformance(LabelSlicedValidationFold,FeedingBehaviorPredicted,[FeedingM, RuminatingM, NothingM],["Feeding", "Ruminating", "Nothing"])
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    s=dt_string+";"+TrainingDataSetName+";"+TrainingDataSetName+";"+str(WindowSize)+";"+str(Fold_i)+";"+str(Vers_i)+";"+str(TotAcc)+";"
    for i in range(len(ConfusionMatr)):
        for j in range(len(ConfusionMatr)):
            s=s+str(ConfusionMatr[i,j])+";"
    fileH=open(ModelFolder+"\\"+ModelName+'_Performance.csv', 'a')
    fileH.write(s+'\n')
    fileH.close()

print('End')