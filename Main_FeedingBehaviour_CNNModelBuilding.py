# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelBuilding.py

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

WindowSizeList=[5,10,20,30,60,90,120,180,300] #sec
# WindowSizeList=[5] #sec
Freq=25 #Hz
FeedingM, RuminatingM, NothingM, DrinkingM = 1, 2, 3, 4
ModelFolder="D:\CowBhave\Models"
FoldN=10

TrainingDataFolder,TrainingDataSetName=[],[]
# TrainingDataFolder.append("D:\\CowBhave\\Data_Exp08_11_2019\\Labeled"),TrainingDataSetName.append("Barn1")
TrainingDataFolder.append("D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"),TrainingDataSetName.append("Barn2")
# TrainingDataFolder.append("D:\\CowBhave\\Mix12"),TrainingDataSetName.append("Mix12")
# TrainingDataFolder.append("D:\\CowBhave\\Pavlovic21\\Labeled"),TrainingDataSetName.append("Barn3")
# TrainingDataFolder.append("D:\\CowBhave\\MixA"),TrainingDataSetName.append("MixA")

Fold_i,Vers_i=0,0
ModelName="NN3"#"RF1"#"SVM1"#

for WindowSize in WindowSizeList:
    print("Interval length " + str(WindowSize) + "sec, model " + ModelName)
    for TrainingDataFolder_i in range(len(TrainingDataFolder)):
        for Fold_i in range(FoldN):
            if ModelName in ["NN1", "NN2", "NN3", "NN4"]:
                if FoldN==1:
                    DataFileName=TrainingDataFolder[TrainingDataFolder_i]+"\\"+'FeedingBehaviour_Training'+"_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5'
                    ModelFileName=ModelFolder+"\\"+ModelName+TrainingDataSetName[TrainingDataFolder_i]+"WS"+str(WindowSize)+"Fold"+str(0)+"V"+str(Vers_i)
                    print('')
                else:
                    DataFileName=TrainingDataFolder[TrainingDataFolder_i]+"\\"+'FeedingBehaviour_Training'+"_WS"+str(WindowSize)+"_F"+str(Freq)+"_Fold"+str(Fold_i+1)+'.h5'
                    ModelFileName=ModelFolder+"\\"+ModelName+TrainingDataSetName[TrainingDataFolder_i]+"WS"+str(WindowSize)+"Fold"+str(Fold_i+1)+"V"+str(Vers_i)
                    print("Fold "+str(Fold_i+1))
                print(DataFileName)
                f = h5py.File(DataFileName, 'r')
                ASlicedTraining = f['AccXYZ'][...]
                LabelSlicedOneHotTraining = f['LabelOneHot'][...]
                LabelSlicedTraining = f['Label'][...]
                f.close()

                epochs, batch_size = 20+50*0, 32

                model=CNNModelDefine(ModelName,ASlicedTraining,LabelSlicedOneHotTraining)
                # print(model.summary())
                train_history = model.fit(ASlicedTraining, LabelSlicedOneHotTraining, epochs=epochs, batch_size=batch_size, verbose=0)
                model.save(ModelFileName)

                print(train_history.history['loss'])
                p=train_history.history['loss']
                with open(ModelFileName+'\\train_history.csv', 'w') as f:
                    for item in p:
                        f.write("%s\n" % item)

            elif ModelName=="RF1" or ModelName=="SVM1":
                if FoldN==1:
                    DataFileName=TrainingDataFolder[TrainingDataFolder_i]+"\\"+"FeedingBehaviour_FeaturesTraining_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5'
                    ModelFileName=ModelFolder+"\\"+ModelName+TrainingDataSetName[TrainingDataFolder_i]+"WS"+str(WindowSize)+"Fold"+str(0)+"V"+str(Vers_i)+".joblib"
                    print('')
                else:
                    DataFileName=TrainingDataFolder[TrainingDataFolder_i]+"\\"+"FeedingBehaviour_FeaturesTraining_WS"+str(WindowSize)+"_F"+str(Freq)+"_Fold"+str(Fold_i+1)+'.h5'
                    ModelFileName=ModelFolder+"\\"+ModelName+TrainingDataSetName[TrainingDataFolder_i]+"WS"+str(WindowSize)+"Fold"+str(Fold_i+1)+"V"+str(Vers_i)+".joblib"
                    print("Fold "+str(Fold_i+1))
                print(DataFileName)
                f = h5py.File(DataFileName, 'r')
                ASlicedFeaturesTraining = f['AccXYZFeatures'][...]
                LabelSlicedTraining = f['Label'][...]
                f.close()

                # print(ASlicedFeaturesTraining[0,:])
                # print(LabelSlicedTraining)
                # print(ASlicedFeaturesTraining.shape)
                # print(LabelSlicedTraining.shape)

                # WindowN=int(WindowSize*Freq)
                # TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label=ReadLabeledDataFiles(TrainingDataFolder[TrainingDataFolder_i],'',Freq)
                # AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, TagNoSlicedT, CowNoSlicedT, TimeStampSlicedT=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
                # AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT=DataBalance(AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT)
                # AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT=AccRotationAugmentation(AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT)
                # LabelSlicedT=numpy.asarray(LabelSlicedT)
                # ASlicedFeaturesTraining,LabelSlicedTraining=[],[]

                # f,l=AccelerationFeatures(AxSlicedT[0],AySlicedT[0],AzSlicedT[0],LabelSlicedT[0],Freq)
                # ASlicedFeaturesTraining=f
                # LabelSlicedTraining=l
                # for i in range(len(AxSlicedT)):
                #     f,l=AccelerationFeatures(AxSlicedT[i],AySlicedT[i],AzSlicedT[i],LabelSlicedT[i],Freq)
                #     ASlicedFeaturesTraining=numpy.vstack((ASlicedFeaturesTraining,f))
                #     LabelSlicedTraining=numpy.vstack((LabelSlicedTraining,l))

                # LabelSlicedTraining=numpy.squeeze(LabelSlicedTraining)

                # print(ASlicedFeaturesTraining[0,:])
                # print(LabelSlicedTraining)
                # print(ASlicedFeaturesTraining.shape)
                # print(LabelSlicedTraining.shape)

                if ModelName=="RF1":
                    clf=RandomForestClassifier(n_estimators=250) #n_estimators=250, Kaler 19; n_estimators=2000, ?max_features=15, Riabof 20; 8, 128, 1 Walton 19;
                elif ModelName=="SVM1":
                    # clf=svm.SVC()
                    # clf=svm.SVC(kernel='poly', degree=8)
                    clf=svm.SVC(kernel='rbf', C=128, gamma=0.05) #Kaler 19; C=128, gamma=0.05, Riaboff 20; Trung 18;
                    # clf=svm.SVC(kernel='sigmoid')

                clf.fit(ASlicedFeaturesTraining,LabelSlicedTraining)
                joblib.dump(clf,ModelFileName)
print('\a')