# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_PrepareDatasetH5PY.py

import h5py
import numpy as np
from FeedingBehavior_NNlib import *
from sklearn.model_selection import KFold

DataFolder,DataSetName=[],[]
mask=""
# DataFolder.append("D:\\CowBhave\\Data_Exp08_11_2019\\Labeled"),DataSetName.append("Barn1"+mask)
# DataFolder.append("D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"),DataSetName.append("Barn2"+mask)
# DataFolder.append("D:\\CowBhave\\Mix12"),DataSetName.append("Mix12"+mask)
DataFolder.append("D:\\CowBhave\\Pavlovic21\\Labeled"),DataSetName.append("Barn3"+mask)
DataFolder.append("D:\\CowBhave\\MixA"),DataSetName.append("MixA"+mask)

WindowSizeList=[5,10,20,30,60,90,120,180,300] #sec
# WindowSizeList=[30,60,90,120,180,300] #sec
Freq=25 #Hz
FoldN=10

for DataFolder_i in range(len(DataFolder)):
    CowNoTotalT=[]
    CowNoTotalV=[]
    FileList=os.listdir(DataFolder[DataFolder_i])
    print("Cow No list")
    for FileName in FileList:
        if FileName.endswith(".csv") and "AccDataLabeled_Tag" in FileName:
            TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label=ReadLabeledDataFiles(DataFolder[DataFolder_i],FileName,Freq)
            CowNoTotalT=list(CowNoTotalT)+list(CowNo)

    for WindowSize in WindowSizeList:
        WindowN=int(WindowSize*Freq)
        # CowNoTotalT=[]
        # CowNoTotalV=[]
    #     fTraining = h5py.File(DataFolder[DataFolder_i]+"\\"+"FeedingBehaviour_Training_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5', 'a')
    #     fValidation=h5py.File(DataFolder[DataFolder_i]+"\\"+"FeedingBehaviour_Validation_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5', 'a')
    #     i=0
    #     for FileName in FileList:
    #         if FileName.endswith(".csv") and "AccDataLabeled_Tag" in FileName:
    #             TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label=ReadLabeledDataFiles(DataFolder[DataFolder_i],FileName,Freq)
    #             AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, TagNoSlicedT, CowNoSlicedT, TimeStampSlicedT=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
    #             AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT=DataBalance(AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT)
    #             AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT=AccRotationAugmentation(AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT)
    #             ASlicedT=numpy.dstack((AxSlicedT,AySlicedT,AzSlicedT))
    #             LabelSlicedT=numpy.asarray(LabelSlicedT) - 1
    #             LabelSlicedOneHotT=to_categorical(LabelSlicedT)
    #             LabelSlicedT=numpy.asarray(LabelSlicedT) + 1
    #             CowNoTotalT=list(CowNoTotalT)+list(CowNoSlicedT)

    #             AxSlicedV, AySlicedV, AzSlicedV, LabelSlicedV, TagNoSlicedV, CowNoSlicedV, TimeStampSlicedV=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
    #             ASlicedV=numpy.dstack((AxSlicedV,AySlicedV,AzSlicedV))
    #             LabelSlicedV=numpy.asarray(LabelSlicedV) - 1
    #             LabelSlicedOneHotV=to_categorical(LabelSlicedV)
    #             LabelSlicedV=numpy.asarray(LabelSlicedV) + 1
    #             CowNoTotalV=list(CowNoTotalV)+list(CowNoSlicedV)

    #             if i==0:
    #                 fTraining.create_dataset("AccXYZ", data=ASlicedT, chunks=True, maxshape=(None,ASlicedT.shape[1],ASlicedT.shape[2]))
    #                 fTraining.create_dataset("LabelOneHot", data=LabelSlicedOneHotT, chunks=True, maxshape=(None,LabelSlicedOneHotT.shape[1]))
    #                 fTraining.create_dataset("Label", data=LabelSlicedT, chunks=True, maxshape=(None,))

    #                 fValidation.create_dataset("AccXYZ", data=ASlicedV, chunks=True, maxshape=(None,ASlicedV.shape[1],ASlicedV.shape[2]))
    #                 fValidation.create_dataset("LabelOneHot", data=LabelSlicedOneHotV, chunks=True, maxshape=(None,LabelSlicedOneHotV.shape[1]))
    #                 fValidation.create_dataset("Label", data=LabelSlicedV, chunks=True, maxshape=(None,))
    #                 i=1
    #             else:
    #                 fTraining["AccXYZ"].resize((fTraining["AccXYZ"].shape[0] + ASlicedT.shape[0]), axis = 0)
    #                 fTraining["AccXYZ"][-ASlicedT.shape[0]:] = ASlicedT
    #                 fTraining["LabelOneHot"].resize((fTraining["LabelOneHot"].shape[0] + LabelSlicedOneHotT.shape[0]), axis = 0)
    #                 fTraining["LabelOneHot"][-LabelSlicedOneHotT.shape[0]:] = LabelSlicedOneHotT
    #                 fTraining["Label"].resize((fTraining["Label"].shape[0] + LabelSlicedT.shape[0]), axis = 0)
    #                 fTraining["Label"][-LabelSlicedT.shape[0]:] = LabelSlicedT

    #                 fValidation["AccXYZ"].resize((fValidation["AccXYZ"].shape[0] + ASlicedV.shape[0]), axis = 0)
    #                 fValidation["AccXYZ"][-ASlicedV.shape[0]:] = ASlicedV
    #                 fValidation["LabelOneHot"].resize((fValidation["LabelOneHot"].shape[0] + LabelSlicedOneHotV.shape[0]), axis = 0)
    #                 fValidation["LabelOneHot"][-LabelSlicedOneHotV.shape[0]:] = LabelSlicedOneHotV
    #                 fValidation["Label"].resize((fValidation["Label"].shape[0] + LabelSlicedV.shape[0]), axis = 0)
    #                 fValidation["Label"][-LabelSlicedV.shape[0]:] = LabelSlicedV
    #     fTraining.close()
    #     fValidation.close()

        # print(ASlicedT.shape)
        # print(LabelSlicedT.shape)
        # plt.plot(ASlicedT[100,:,0],'b')

        # f = h5py.File(DataFolder[DataFolder_i]+"\\"+"FeedingBehaviour_Training_WS"+str(WindowSize)+"_F"+str(Freq)+'.h5', 'r')
        # ASlicedTraining = f['AccXYZ'][...]
        # LabelSlicedOneHotTraining = f['LabelOneHot'][...]
        # LabelSlicedTraining = f['Label'][...]
        # f.close()

        # print(ASlicedTraining.shape)
        # print(LabelSlicedTraining.shape)
        # plt.plot(ASlicedTraining[100,:,0],'r')
        # plt.show()

        CowNoList=numpy.unique(list(CowNoTotalT))
        print('Cow no list')
        print(str(len(CowNoList))+" cows: "+str(CowNoList))
        kf = KFold(n_splits=FoldN)

        #Training
        print("Training folds")
        Fold_i=0
        for train_index, test_index in kf.split(CowNoList):
            Fold_i=Fold_i+1
            jt,jv=0,0
            DataFileName=DataFolder[DataFolder_i]+"\\"+"FeedingBehaviour_Training_WS"+str(WindowSize)+"_F"+str(Freq)+"_Fold"+str(Fold_i)+'.h5'
            fTrainingFold = h5py.File(DataFileName, 'a')
            DataFileName=DataFolder[DataFolder_i]+"\\"+"FeedingBehaviour_Validation_WS"+str(WindowSize)+"_F"+str(Freq)+"_Fold"+str(Fold_i)+'.h5'
            fValidationFold=h5py.File(DataFileName, 'a')
            print("Interval length " + str(WindowSize) + "sec, " + "Fold "+str(Fold_i)+", training="+str(CowNoList[train_index])+", validation="+str(CowNoList[test_index]))
            
            for FileName in FileList:
                if FileName.endswith(".csv") and "AccDataLabeled_Tag" in FileName:
                    TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label=ReadLabeledDataFiles(DataFolder[DataFolder_i],FileName,Freq)
                    if CowNo[0] in CowNoList[train_index]:
                        AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, TagNoSlicedT, CowNoSlicedT, TimeStampSlicedT=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
                        AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT=DataBalance(AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT)
                        AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT=AccRotationAugmentation(AxSlicedT, AySlicedT, AzSlicedT, LabelSlicedT, CowNoSlicedT)
                        ASlicedT=numpy.dstack((AxSlicedT,AySlicedT,AzSlicedT))
                        LabelSlicedT=numpy.asarray(LabelSlicedT) - 1
                        LabelSlicedOneHotT=to_categorical(LabelSlicedT)
                        LabelSlicedT=numpy.asarray(LabelSlicedT) + 1
                        if jt==0:
                            fTrainingFold.create_dataset("AccXYZ", data=ASlicedT, chunks=True, maxshape=(None,ASlicedT.shape[1],ASlicedT.shape[2]))
                            fTrainingFold.create_dataset("LabelOneHot", data=LabelSlicedOneHotT, chunks=True, maxshape=(None,LabelSlicedOneHotT.shape[1]))
                            fTrainingFold.create_dataset("Label", data=LabelSlicedT, chunks=True, maxshape=(None,))
                            jt=1
                        else:
                            fTrainingFold["AccXYZ"].resize((fTrainingFold["AccXYZ"].shape[0] + ASlicedT.shape[0]), axis = 0)
                            fTrainingFold["AccXYZ"][-ASlicedT.shape[0]:] = ASlicedT
                            fTrainingFold["LabelOneHot"].resize((fTrainingFold["LabelOneHot"].shape[0] + LabelSlicedOneHotT.shape[0]), axis = 0)
                            fTrainingFold["LabelOneHot"][-LabelSlicedOneHotT.shape[0]:] = LabelSlicedOneHotT
                            fTrainingFold["Label"].resize((fTrainingFold["Label"].shape[0] + LabelSlicedT.shape[0]), axis = 0)
                            fTrainingFold["Label"][-LabelSlicedT.shape[0]:] = LabelSlicedT
                    
                    elif CowNo[0] in CowNoList[test_index]:
                        AxSlicedV, AySlicedV, AzSlicedV, LabelSlicedV, TagNoSlicedV, CowNoSlicedV, TimeStampSlicedV=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
                        ASlicedV=numpy.dstack((AxSlicedV,AySlicedV,AzSlicedV))
                        LabelSlicedV=numpy.asarray(LabelSlicedV) - 1
                        LabelSlicedOneHotV=to_categorical(LabelSlicedV)
                        LabelSlicedV=numpy.asarray(LabelSlicedV) + 1
                        if jv==0:
                            fValidationFold.create_dataset("AccXYZ", data=ASlicedV, chunks=True, maxshape=(None,ASlicedV.shape[1],ASlicedV.shape[2]))
                            fValidationFold.create_dataset("LabelOneHot", data=LabelSlicedOneHotV, chunks=True, maxshape=(None,LabelSlicedOneHotV.shape[1]))
                            fValidationFold.create_dataset("Label", data=LabelSlicedV, chunks=True, maxshape=(None,))
                            jv=1
                        else:
                            fValidationFold["AccXYZ"].resize((fValidationFold["AccXYZ"].shape[0] + ASlicedV.shape[0]), axis = 0)
                            fValidationFold["AccXYZ"][-ASlicedV.shape[0]:] = ASlicedV
                            fValidationFold["LabelOneHot"].resize((fValidationFold["LabelOneHot"].shape[0] + LabelSlicedOneHotV.shape[0]), axis = 0)
                            fValidationFold["LabelOneHot"][-LabelSlicedOneHotV.shape[0]:] = LabelSlicedOneHotV
                            fValidationFold["Label"].resize((fValidationFold["Label"].shape[0] + LabelSlicedV.shape[0]), axis = 0)
                            fValidationFold["Label"][-LabelSlicedV.shape[0]:] = LabelSlicedV

            fTrainingFold.close()
            fValidationFold.close()
        
print('\a')