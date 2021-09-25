# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelDevelop.py

from FeedingBehavior_NNlib import *
import fnmatch, os
from sklearn.model_selection import train_test_split
import numpy
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
# from tensorflow.keras.utils import to_categorical
# import tensorflow
import matplotlib.pyplot as plt

# FoldTagList=[7, 39, 55, 73, 75]
FoldCowNoList=[336, 414, 649, 683, 686, 707, 715]
FoldCowNoList=[[649, 683, 686, 715],
               [414, 683, 686, 715],
               [414, 649, 686, 715],
               [414, 649, 683, 715],
               [414, 649, 683, 686]]

WindowSize=10 #sec
Freq=25 #Hz
WindowN=int(WindowSize*Freq)
print("Interval length " + str(WindowSize) + "sec")

print('Reading')
# DataFolder="D:\CowBhave\Data_Exp08_11_2019\Labeled"
DataFolder="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"
DataSetName="Barn2"
ModelName="NN1"
ModelFolder="D:\CowBhave\Models"
TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label = ReadLabeledDataFiles(DataFolder,'AccDataLabeled_Tag')
# FileName="A_Exp08_11_2019_LabeledAcc.csv"

print('Slicing')
AxSliced, AySliced, AzSliced, LabelSliced, TagNoSliced, CowNoSliced, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 0.5, TagNo, CowNo, TimeStamp)

print('Balancing')
AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced=DataBalance(AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced, TimeStampSliced)
AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced=AccAugmentation(AxSliced, AySliced, AzSliced, LabelSliced, CowNoSliced, 5)
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

epochs, batch_size = 50, 32
for Fold_i in range(len(FoldCowNoList[0])*0+1):
    Vers_i=0
    print("Cows "+str(FoldCowNoList[Fold_i])+", fold "+str(Fold_i)+", version "+str(Vers_i))
    AxSlicedFold,AySlicedFold,AzSlicedFold,LabelSlicedFold=[],[],[],[]
    for i in range(len(LabelSliced)):        
        if CowNoSliced[i] in FoldCowNoList[Fold_i]:
            AxSlicedFold.append(AxSliced[i])
            AySlicedFold.append(AySliced[i])
            AzSlicedFold.append(AzSliced[i])
            LabelSlicedFold.append(LabelSliced[i])

    print('Training')

    if ModelName in ["NN1", "NN2", "NN3", "NN4"]:
        ASlicedFold=numpy.dstack((AxSlicedFold,AySlicedFold,AzSlicedFold))
        # print(ASlicedFold.shape)
        trainX, testX, trainy, testy=train_test_split(ASlicedFold,LabelSlicedFold,test_size=0.2, random_state=0, stratify=LabelSlicedFold)
        print("Test sample number " + str(len(testX)) + ", train sample number " + str(len(trainX)))
        trainy =numpy.asarray(trainy) - 1
        testy = numpy.asarray(testy) - 1
        trainy = to_categorical(trainy)
        testy = to_categorical(testy)

        model=CNNModelDefine(ModelName,trainX,trainy)
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)
        _, score = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        print('>#%d: %.3f' % (Fold_i+1, score* 100.0))
    elif ModelName=="RF1":
        

    model.save(ModelFolder+"\\"+ModelName+DataSetName+"WS"+str(WindowSize)+"Fold"+str(Fold_i)+"V"+str(Vers_i))

print('End')