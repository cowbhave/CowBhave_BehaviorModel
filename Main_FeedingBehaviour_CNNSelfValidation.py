# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNSelfValidation.py

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

ModelFolder="D:\CowBhave\Models"
WindowSizeList=[5,10,20,30,60,90,120,180,300] #sec
# WindowSizeList=[20] #sec
Freq=25 #Hz
FeedingM, RuminatingM, NothingM, DrinkingM = 1, 2, 3, 4
FoldN=10

TrainingDataFolder,TrainingDataSetName=[],[]
# TrainingDataFolder.append("D:\\CowBhave\\Data_Exp08_11_2019\\Labeled"),TrainingDataSetName.append("Barn1")
TrainingDataFolder.append("D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"),TrainingDataSetName.append("Barn2")
# TrainingDataFolder.append("D:\\CowBhave\\Mix12"),TrainingDataSetName.append("Mix12")
# TrainingDataFolder.append("D:\\CowBhave\\Pavlovic21\\Labeled"),TrainingDataSetName.append("Barn3")
# TrainingDataFolder.append("D:\\CowBhave\\MixA"),TrainingDataSetName.append("MixA")

Vers_i=0
ModelName="NN1"#"RF1"#"SVM1"#
for WindowSize in WindowSizeList:
    # WindowN=int(WindowSize*Freq)
    for TrainingDataSet_i in range(len(TrainingDataSetName)):
        for Fold_i in range(FoldN):
            # print(ModelName + ", WS=" + str(WindowSize) + ", " + TrainingDataSetName[TrainingDataSet_i] + ", " + TrainingDataSetName[TrainingDataSet_i] +", Fold "+str(Fold_i))
            if ModelName in ["NN1", "NN2", "NN3", "NN4"]:
                DataFileName=TrainingDataFolder[TrainingDataSet_i]+"\\"+'FeedingBehaviour_Validation'+"_WS"+str(WindowSize)+"_F"+str(Freq)+"_Fold"+str(Fold_i+1)+'.h5'
                print(DataFileName)
                f = h5py.File(DataFileName, 'r')
                ASlicedValidation = f['AccXYZ'][...]
                LabelSlicedValidation = f['Label'][...]
                f.close()
                model=tensorflow.keras.models.load_model(ModelFolder+"\\"+ModelName+TrainingDataSetName[TrainingDataSet_i]+"WS"+str(WindowSize)+"Fold"+str(Fold_i+1)+"V"+str(Vers_i))

                FeedingBehaviorPredictedP=model.predict(ASlicedValidation)
                FeedingBehaviorPredicted=list()
                for i in range(len(FeedingBehaviorPredictedP)):
                    m,k=MaxInd(FeedingBehaviorPredictedP[i])
                    FeedingBehaviorPredicted.append(k+1)

            elif ModelName=="RF1" or ModelName=="SVM1":
                DataFileName=TrainingDataFolder[TrainingDataSet_i]+"\\"+"FeedingBehaviour_FeaturesValidation_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5'
                f = h5py.File(DataFileName, 'r')
                print(DataFileName)
                ASlicedFeaturesValidation = f['AccXYZFeatures'][...]
                LabelSlicedValidation = f['Label'][...]
                f.close()

                # WindowN=int(WindowSize*Freq)
                # TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label=ReadLabeledDataFiles(ValidationDataFolder[ValidationDataSet_i],'',Freq)
                # AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, TagNoSlicedT, CowNoSlicedT, TimeStampSlicedT=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
                # AxSlicedT=numpy.squeeze(AxSlicedT)
                # AySlicedT=numpy.squeeze(AySlicedT)
                # AzSlicedT=numpy.squeeze(AzSlicedT)
                # LabelSlicedT=numpy.squeeze(LabelSlicedT)

                # f,l=AccelerationFeatures(AxSlicedT[0],AySlicedT[0],AzSlicedT[0],LabelSlicedT[0],Freq)
                # ASlicedFeaturesValidation=f
                # LabelSlicedValidation=l
                # for i in range(len(AxSlicedT)):
                #     f,l=AccelerationFeatures(AxSlicedT[i],AySlicedT[i],AzSlicedT[i],LabelSlicedT[i],Freq)
                #     ASlicedFeaturesValidation=numpy.vstack((ASlicedFeaturesValidation,f))
                #     LabelSlicedValidation=numpy.vstack((LabelSlicedValidation,l))

                # LabelSlicedValidation=numpy.squeeze(LabelSlicedValidation)

            # print(ASlicedFeaturesValidation)
            # print(LabelSlicedValidation)
            # print(ASlicedFeaturesValidation.shape)
            # print(LabelSlicedValidation.shape)


                clf = joblib.load(ModelFolder+"\\"+ModelName+TrainingDataSetName[TrainingDataSet_i]+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i)+".joblib")
                FeedingBehaviorPredicted=clf.predict(ASlicedFeaturesValidation)+1*0    

            FeedingBehaviorPredicted=scipy.signal.medfilt(FeedingBehaviorPredicted,5)
            PerfVect, ConfusionMatr, TotAcc=PresentPerformance(LabelSlicedValidation,FeedingBehaviorPredicted,[FeedingM, RuminatingM, NothingM],["Feeding", "Ruminating", "Nothing"])
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H%M%S")
            s=dt_string+";"+ModelName+";"+TrainingDataSetName[TrainingDataSet_i]+";"+TrainingDataSetName[TrainingDataSet_i]+";"+str(WindowSize)+";"+str(Fold_i+1)+";"+str(Vers_i)+";"+str(TotAcc)+";"
            for i in range(len(ConfusionMatr)):
                for j in range(len(ConfusionMatr)):
                    s=s+str(ConfusionMatr[i,j])+";"
            fileH=open(ModelFolder+"\\"+'Performance.csv', 'a')
            fileH.write(s+'\n')
            fileH.close()

print('\a')