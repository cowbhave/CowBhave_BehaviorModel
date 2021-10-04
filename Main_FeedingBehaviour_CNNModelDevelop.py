# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelDevelop.py

from FeedingBehavior_NNlib import *
import fnmatch, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# DataFolder="D:\CowBhave\Data_Exp08_11_2019\Labeled"
DataFolder="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"
DataSetName="Barn2"
ModelName="RF1"#"NN1"#
ModelFolder="D:\CowBhave\Models"
print('Reading')
TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label = ReadLabeledDataFiles(DataFolder,'AccDataLabeled_Tag')
CowNoList=numpy.unique(CowNo)
print('Cow no list')
print(CowNoList)
print('Slicing')
AxSliced, AySliced, AzSliced, LabelSliced, TagNoSliced, CowNoSliced, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)

print('Balancing')
# AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced=DataBalance(AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced, TimeStampSliced)
# AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced=AccAugmentation(AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced, 5)
# # for i in range(0,10):
# #     plt.scatter(range(WindowN), AxSliced[i])
# #     plt.scatter(range(WindowN), AySliced[i])
# #     plt.scatter(range(WindowN), AzSliced[i])
# # for i in range(0,10):
# #     plt.scatter(range(WindowN*(i-1),WindowN*i), AxSliced[i], color='black')
# #     plt.scatter(range(WindowN*(i-1),WindowN*i), AySliced[i], color='blue')
# #     plt.scatter(range(WindowN*(i-1),WindowN*i), AzSliced[i], color='green')
# # # plt.figure()
# # # for i in range(0,50):
# # #     plt.scatter(range(WindowN*(i-1),WindowN*i), AxSliced[i], color='black')
# # #     plt.scatter(range(WindowN*(i-1),WindowN*i), AySliced[i], color='blue')
# # #     plt.scatter(range(WindowN*(i-1),WindowN*i), AzSliced[i], color='green')
# # plt.show()

kf = KFold(n_splits=5)
epochs, batch_size = 20, 32
Fold_i=0
# for Fold_i in range(len(FoldCowNoList[0])*0+1):
for train_index, test_index in kf.split(CowNoList):
    FoldCowNoList=CowNoList[train_index]
    Fold_i=Fold_i+1
    Vers_i=0
    # print("Cows "+str(FoldCowNoList[Fold_i])+", fold "+str(Fold_i)+", version "+str(Vers_i))
    print("Cows "+str(FoldCowNoList)+", fold "+str(Fold_i)+", version "+str(Vers_i))
    AxSlicedTrainFold,AySlicedTrainFold,AzSlicedTrainFold,LabelSlicedTrainFold=[],[],[],[]
    AxSlicedTestFold,AySlicedTestFold,AzSlicedTestFold,LabelSlicedTestFold=[],[],[],[]
    for i in range(len(LabelSliced)):        
        # if CowNoSliced[i] in FoldCowNoList[Fold_i]:
        if CowNoSliced[i] in FoldCowNoList:
            AxSlicedTrainFold.append(AxSliced[i])
            AySlicedTrainFold.append(AySliced[i])
            AzSlicedTrainFold.append(AzSliced[i])
            LabelSlicedTrainFold.append(LabelSliced[i])
        else:
            AxSlicedTestFold.append(AxSliced[i])
            AySlicedTestFold.append(AySliced[i])
            AzSlicedTestFold.append(AzSliced[i])
            LabelSlicedTestFold.append(LabelSliced[i])    
    
    if ModelName in ["NN1", "NN2", "NN3", "NN4"]:
        print('Training')
        ASlicedTrainFold=numpy.dstack((AxSlicedTrainFold,AySlicedTrainFold,AzSlicedTrainFold))
        # print(ASlicedTrainFold.shape)
        trainX, testX, trainy, testy=train_test_split(ASlicedTrainFold,LabelSlicedTrainFold,test_size=0.2, random_state=0, stratify=LabelSlicedTrainFold)
        print("Test sample number " + str(len(testX)) + ", train sample number " + str(len(trainX)))
        trainy =numpy.asarray(trainy) - 1
        testy = numpy.asarray(testy) - 1
        trainy = to_categorical(trainy)
        testy = to_categorical(testy)

        model=CNNModelDefine(ModelName,trainX,trainy)
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)
        _, score = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        print('>#%d: %.3f' % (Fold_i+1, score* 100.0))
        model.save(ModelFolder+"\\"+ModelName+DataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))

        print('Validating')
        ASlicedTestFold=numpy.dstack((AxSlicedTestFold,AySlicedTestFold,AzSlicedTestFold))
        FeedingBehaviorPredictedP=model.predict(ASlicedTestFold)
        FeedingBehaviorPredicted=list()
        for i in range(len(FeedingBehaviorPredictedP)):
            m,k=MaxInd(FeedingBehaviorPredictedP[i])
            FeedingBehaviorPredicted.append(k+1)

        FeedingBehaviorPredicted=scipy.signal.medfilt(FeedingBehaviorPredicted,5)

    elif ModelName=="RF1":
        print('Training')
        # AxSlicedTrainFold_Mean=numpy.mean(AxSlicedTrainFold, axis=1)
        # AySlicedTrainFold_Mean=numpy.mean(AySlicedTrainFold, axis=1)
        # AzSlicedTrainFold_Mean=numpy.mean(AzSlicedTrainFold, axis=1)
        # FeaturesTrainFold=numpy.dstack((AxSlicedTrainFold_Mean,AySlicedTrainFold_Mean,AzSlicedTrainFold_Mean))
        # FeaturesTrainFold=numpy.squeeze(FeaturesTrainFold)
        # LabelSlicedTrainFold=numpy.squeeze(LabelSlicedTrainFold)
        FeaturesTrainFold,LabelSlicedTrainFold=RandomForestFeatures(AxSlicedTrainFold,AySlicedTrainFold,AzSlicedTrainFold,LabelSlicedTrainFold)

        trainX, testX, trainy, testy=train_test_split(FeaturesTrainFold,LabelSlicedTrainFold,test_size=0.2, random_state=0, stratify=LabelSlicedTrainFold)
        print("Test sample number " + str(len(testX)) + ", train sample number " + str(len(trainX)))
        trainy=trainy - 1
        testy = testy - 1

        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(trainX,trainy)
        joblib.dump(clf, ModelFolder+"\\"+ModelName+DataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i)+".joblib")
        # clf = joblib.load("./random_forest.joblib")

        print('Validating')
        # AxSlicedTestFold_Mean=numpy.mean(AxSlicedTestFold, axis=1)
        # AySlicedTestFold_Mean=numpy.mean(AySlicedTestFold, axis=1)
        # AzSlicedTestFold_Mean=numpy.mean(AzSlicedTestFold, axis=1)
        # FeaturesTestFold=numpy.dstack((AxSlicedTestFold_Mean,AySlicedTestFold_Mean,AzSlicedTestFold_Mean))
        # FeaturesTestFold=numpy.squeeze(FeaturesTestFold)
        # LabelSlicedTestFold=numpy.squeeze(LabelSlicedTestFold)
        FeaturesTestFold,LabelSlicedTestFold=RandomForestFeatures(AxSlicedTestFold,AySlicedTestFold,AzSlicedTestFold,LabelSlicedTestFold)

        FeedingBehaviorPredicted=clf.predict(FeaturesTestFold)+1

    # print(len(LabelSlicedTestFold))
    # print(len(FeedingBehaviorPredicted))
    PerfVect, ConfusionMatr=PresentPerformance(LabelSlicedTestFold,FeedingBehaviorPredicted,[FeedingM, RuminatingM, NothingM],["Feeding", "Ruminating", "Nothing"])
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    s=dt_string+";"+DataSetName+";"+str(WindowSize)+";"+str(Fold_i)+";"+str(Vers_i)+";"
    for i in range(len(ConfusionMatr)):
        for j in range(len(ConfusionMatr)):
            s=s+str(ConfusionMatr[i,j])+";"
    fileH=open(ModelFolder+"\\"+ModelName+'_Performance.csv', 'a')
    fileH.write(s)
    fileH.close()


print('End')