# .\venv\Scripts\activate
# cd C:\Users\03138529\Dropbox\Luke\CowBhave
# python Main_FeedingBehaviour_CNNModelPredict.py
from FeedingBehavior_NNlib import *
import fnmatch, os
from sklearn.model_selection import train_test_split
import numpy
import scipy.signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import tensorflow
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

WindowSize=20 #sec
Freq=25 #Hz
WindowN=int(WindowSize*Freq)

DataFolder="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled"
# DataFolder="D:\CowBhave\Data_Exp08_11_2019\Labeled"
ModelFolder="D:\\CowBhave\\Data_Exp15_02_2021\\Labeled\\CNN_Maaninka"
# ModelFolder="D:\\CowBhave\\Data_Exp08_11_2019\\Labeled\\CNN_Viikki"
model=tensorflow.keras.models.load_model(ModelFolder)
# plot_model(model, to_file=DataFolder+'\\model_plot.png', show_shapes=True, show_layer_names=True)
tensorflow.keras.utils.plot_model(model, to_file='Model1.png')
# FileList=os.listdir(DataFolder)
# for FileName in FileList:
#     if FileName.endswith(".csv") and 'AccDataLabeled_Tag' in FileName:
#         TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label = ReadLabeledDataFiles(DataFolder,FileName)
#         AxSliced, AySliced, AzSliced, LabelSliced, TagNoSliced, CowNoSliced, TimeStampSliced=LabeledDataSlicing(Ax, Ay, Az, Label, WindowN, 1, TagNo, CowNo, TimeStamp)
#         ASliced=numpy.dstack((AxSliced,AySliced,AzSliced))
#         FeedingBehaviorPredictedP=model.predict(ASliced)
#         FeedingBehaviorPredicted=list()
#         for i in range(len(FeedingBehaviorPredictedP)):
#             m,k=MaxInd(FeedingBehaviorPredictedP[i])
#             FeedingBehaviorPredicted.append(k+1)

#         FeedingBehaviorPredicted=scipy.signal.medfilt(FeedingBehaviorPredicted,5)

#         T=numpy.column_stack([TagNoSliced,CowNoSliced,TimeStampSliced,FeedingBehaviorPredicted])
#         numpy.savetxt(DataFolder+"\\FeedingBehaviorPredicted_Cow"+str(CowNo[0])+"_"+FileName[(len(FileName)-14):(len(FileName)-4)]+".csv", T, delimiter=';', fmt="%s")
