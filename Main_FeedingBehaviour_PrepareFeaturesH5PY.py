# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_PrepareFeaturesH5PY.py

import h5py
import numpy
from FeedingBehavior_NNlib import AccelerationFeatures

WindowSizeList=[5,10,20,30,60,90,120,180,300] #sec
# WindowSizeList=[20] #sec
Freq=25 #Hz

DataFolder,DataSetName=[],[]
mask=""
DataFolder.append("D:\\CowBhave\\Data_Exp08_11_2019\\Labeled"),DataSetName.append("Barn1"+mask)
DataFolder.append("D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"),DataSetName.append("Barn2"+mask)
DataFolder.append("D:\\CowBhave\\Mix12"),DataSetName.append("Mix12"+mask)
DataFolder.append("D:\\CowBhave\\Pavlovic21\\Labeled"),DataSetName.append("Barn3"+mask)
DataFolder.append("D:\\CowBhave\\MixA"),DataSetName.append("MixA"+mask)

for DataFolder_i in range(len(DataFolder)):
    for WindowSize in WindowSizeList:
        DataFileName=DataFolder[DataFolder_i]+"\\"+'FeedingBehaviour_Training_WS'+str(WindowSize)+"_F"+str(Freq)+'.h5'
        print(DataFileName)
        f = h5py.File(DataFileName, 'r')
        ASlicedTraining = f['AccXYZ'][...]
        LabelSlicedTraining = f['Label'][...]
        f.close()

        fFeaturesTraining = h5py.File(DataFolder[DataFolder_i]+"\\"+"FeedingBehaviour_FeaturesTraining_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5', 'a')
        i=0
        for j in range(0,ASlicedTraining.shape[0]-2,2):
            aSliced=ASlicedTraining[j]
            axSliced=aSliced[:,0]
            aySliced=aSliced[:,1]
            azSliced=aSliced[:,2]
            labelSliced=LabelSlicedTraining[j]
            AccXYZFeatures1,LabelFeatures1=AccelerationFeatures(axSliced,aySliced,azSliced,labelSliced,Freq)
            aSliced=ASlicedTraining[j+1]
            axSliced=aSliced[:,0]
            aySliced=aSliced[:,1]
            azSliced=aSliced[:,2]
            labelSliced=LabelSlicedTraining[j]
            AccXYZFeatures2,LabelFeatures2=AccelerationFeatures(axSliced,aySliced,azSliced,labelSliced,Freq)

            AccXYZFeatures=numpy.vstack((AccXYZFeatures1,AccXYZFeatures2))
            LabelFeatures=numpy.vstack((LabelFeatures1,LabelFeatures2))
            AccXYZFeatures=numpy.squeeze(AccXYZFeatures)
            LabelFeatures=numpy.squeeze(LabelFeatures)
            # print(numpy.squeeze(AccXYZFeatures))
            # print(numpy.squeeze(LabelFeatures))

            if i==0:
                fFeaturesTraining.create_dataset("AccXYZFeatures", data=AccXYZFeatures, chunks=True, maxshape=(None,AccXYZFeatures.shape[1]))
                fFeaturesTraining.create_dataset("Label", data=LabelFeatures, chunks=True, maxshape=(None,))
                i=1
            else:
                fFeaturesTraining["AccXYZFeatures"].resize((fFeaturesTraining["AccXYZFeatures"].shape[0] + AccXYZFeatures.shape[0]), axis = 0)
                fFeaturesTraining["AccXYZFeatures"][-AccXYZFeatures.shape[0]:] = AccXYZFeatures
                fFeaturesTraining["Label"].resize((fFeaturesTraining["Label"].shape[0] + LabelFeatures.shape[0]), axis = 0)
                fFeaturesTraining["Label"][-LabelFeatures.shape[0]:] = LabelFeatures
        fFeaturesTraining.close()

        # print(AccXYZFeatures)
        # print(LabelFeatures)
        # print(AccXYZFeatures.shape)
        # print(LabelFeatures.shape)


        DataFileName=DataFolder[DataFolder_i]+"\\"+'FeedingBehaviour_Validation_WS'+str(WindowSize)+"_F"+str(Freq)+'.h5'
        f = h5py.File(DataFileName, 'r')
        print(DataFileName)
        ASlicedValidation = f['AccXYZ'][...]
        LabelSlicedValidation = f['Label'][...]
        f.close()

        fFeaturesValidation=h5py.File(DataFolder[DataFolder_i]+"\\"+"FeedingBehaviour_FeaturesValidation_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5', 'a')
        i=0
        for j in range(0,ASlicedValidation.shape[0]-2,2):
            aSliced=ASlicedValidation[j]
            axSliced=aSliced[:,0]
            aySliced=aSliced[:,1]
            azSliced=aSliced[:,2]
            labelSliced=LabelSlicedValidation[j]
            AccXYZFeatures1,LabelFeatures1=AccelerationFeatures(axSliced,aySliced,azSliced,labelSliced,Freq)
            aSliced=ASlicedValidation[j+1]
            axSliced=aSliced[:,0]
            aySliced=aSliced[:,1]
            azSliced=aSliced[:,2]
            labelSliced=LabelSlicedValidation[j]
            AccXYZFeatures2,LabelFeatures2=AccelerationFeatures(axSliced,aySliced,azSliced,labelSliced,Freq)

            AccXYZFeatures=numpy.vstack((AccXYZFeatures1,AccXYZFeatures2))
            LabelFeatures=numpy.vstack((LabelFeatures1,LabelFeatures2))

            if i==0:
                fFeaturesValidation.create_dataset("AccXYZFeatures", data=AccXYZFeatures, chunks=True, maxshape=(None,AccXYZFeatures.shape[1]))
                fFeaturesValidation.create_dataset("Label", data=LabelFeatures, chunks=True, maxshape=(None,1))
                i=1
            else:
                fFeaturesValidation["AccXYZFeatures"].resize((fFeaturesValidation["AccXYZFeatures"].shape[0] + AccXYZFeatures.shape[0]), axis = 0)
                fFeaturesValidation["AccXYZFeatures"][-AccXYZFeatures.shape[0]:] = AccXYZFeatures
                fFeaturesValidation["Label"].resize((fFeaturesValidation["Label"].shape[0] + LabelFeatures.shape[0]), axis = 0)
                fFeaturesValidation["Label"][-LabelFeatures.shape[0]:] = LabelFeatures
        fFeaturesValidation.close()
