# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_PrepareDatasetH5PY.py

import h5py
import numpy as np
from FeedingBehavior_NNlib import *

DataFolder="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"
# DataFolder="D:\\CowBhave\\Data_Exp08_11_2019\\Labeled"
# DataFolder="D:\\CowBhave\\Pavlovic21\\Labeled"

WindowSize=10 #sec
Freq=25 #Hz
WindowN=int(WindowSize*Freq)

FileList=os.listdir(DataFolder)
i=0
with h5py.File(DataFolder+"\\"+'FeedingBehaviour'+"WS"+str(WindowSize)+"F"+str(Freq)+'.h5', 'a') as hf:
    for FileName in FileList:
        if FileName.endswith(".csv") and "AccDataLabeled_Tag" in FileName:
            print(FileName)
            TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label=ReadLabeledDataFile(DataFolder,FileName)

            Ax, Ay, Az=AccelerationFiltering(Ax, Ay, Az)
            Ax, Ay, Az, TimeStamp, Label, TagNo, CowNo=AccelerationSamplingFitting(Ax, Ay, Az, TimeStamp, Label, TagNo, CowNo, Freq)
            AxSliced, AySliced, AzSliced, LabelSliced, TagNoSliced, CowNoSliced, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)
            print('Balancing')
            AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced=DataBalance(AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced)
            AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced=AccAugmentation(AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced, 5)
            ASliced=numpy.dstack((AxSliced,AySliced,AzSliced))
            LabelSliced=numpy.asarray(LabelSliced) - 1
            LabelSlicedOneHot=to_categorical(LabelSliced)

            if i==0:
                hf.create_dataset("AccXYZ", data=ASliced, chunks=True, maxshape=(None,ASliced.shape[1],ASliced.shape[2]))
                hf.create_dataset("LabelOneHot", data=LabelSlicedOneHot, chunks=True, maxshape=(None,LabelSlicedOneHot.shape[1]))
                hf.create_dataset("Label", data=LabelSliced, chunks=True, maxshape=(None,))
                i=1
            else:
                hf["AccXYZ"].resize((hf["AccXYZ"].shape[0] + ASliced.shape[0]), axis = 0)
                hf["AccXYZ"][-ASliced.shape[0]:] = ASliced
                hf["LabelOneHot"].resize((hf["LabelOneHot"].shape[0] + LabelSlicedOneHot.shape[0]), axis = 0)
                hf["LabelOneHot"][-LabelSlicedOneHot.shape[0]:] = LabelSlicedOneHot
                hf["Label"].resize((hf["Label"].shape[0] + LabelSliced.shape[0]), axis = 0)
                hf["Label"][-LabelSliced.shape[0]:] = LabelSliced
