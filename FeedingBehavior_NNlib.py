import csv
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D, LSTM, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
import tensorflow
import fnmatch, os
import math
import datetime
from datetime import datetime
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
# python FeedingBehavior_NNlib.py

def CNNModelDefine(ModelName,trainX,trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    if ModelName=="NN1": #https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif ModelName=="NN2": #https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
        n_length = 32
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))    
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif ModelName=="NN3": #Ordóñez 16, https://doi.org/10.3390/s16010115, https://github.com/STRCWearlab/DeepConvLSTM
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape = (n_timesteps, n_features)))
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(Dropout(0.5))
        model.add(LSTM(128, activation='tanh', return_sequences=True))
        model.add(LSTM(128, activation='tanh'))
        model.add(Dense(n_outputs, activation='softmax', dtype='float32'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy", "categorical_accuracy"])
    elif ModelName=="NN4": #Pavlovic 21, 
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=32, strides=1, activation='relu', input_shape = (n_timesteps, n_features)))
        model.add(Conv1D(filters=64, kernel_size=32, strides=2, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=32, strides=2, activation='relu'))
        model.add(Conv1D(filters=512, kernel_size=1, strides=1, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy", "categorical_accuracy"])
    else:
        print()
    return model

def ReadLabeledDataFiles(DataFolder,Mask,RefFreq):
    FileList=os.listdir(DataFolder)
    Ax=[]
    Ay=[]
    Az=[]
    Label=[]
    TimeStamp=[]
    CowNo=[]
    TagNo=[]
    for FileName in FileList:
        if FileName.endswith(".csv") and Mask in FileName:
            print(FileName)
            k=FileName.find('_Tag')+4
            m=FileName.find('_Cow')
            l=FileName.find('_',m+2)
            tagno=FileName[k:m]
            cowno=FileName[m+4:l]

            ax=[]
            ay=[]
            az=[]
            label=[]
            timeStamp=[]
            cowNo=[]
            tagNo=[]
            with open(DataFolder+"\\"+FileName, mode ='r')as csvfile:
                csvreader = csv.reader(csvfile, delimiter=";")
                rows=[]
                for row in csvreader:
                    rows.append(row)

            for i in range(len(rows)):
                cowNo.append(cowno)
                tagNo.append(tagno)
                # TimeStamp.append(rows[i][0])
                timeStamp.append(datetime.strptime(rows[i][0],"%Y-%m-%dT%H:%M:%S.%f").timestamp())
                ax.append(int(rows[i][1]))
                ay.append(int(rows[i][2]))
                az.append(int(rows[i][3]))
                label.append(int(rows[i][4]))

            ax, ay, az=AccelerationFiltering(ax, ay, az)
            ax, ay, az, timeStamp, label, tagNo, cowNo=AccelerationSamplingFitting(ax, ay, az, timeStamp, label, tagNo, cowNo, RefFreq)

            CowNo=list(CowNo)+list(cowNo)
            TagNo=list(TagNo)+list(tagNo)
            TimeStamp=list(TimeStamp)+list(timeStamp)
            Label=list(Label)+list(label)
            Ax=list(Ax)+list(ax)
            Ay=list(Ay)+list(ay)
            Az=list(Az)+list(az)
    
    Label=numpy.asarray(Label)
    CowNo=numpy.asarray(CowNo)
    TagNo=numpy.asarray(TagNo)
    return TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label

def ReadLabeledDataFile(DataFolder,FileName):
    Ax=[]
    Ay=[]
    Az=[]
    Label=[]
    TimeStamp=[]
    CowNo=[]
    TagNo=[]

    k=FileName.find('_Tag')+4
    m=FileName.find('_Cow')
    l=FileName.find('_',m+2)
    tagno=FileName[k:m]
    cowno=FileName[m+4:l]

    with open(DataFolder+"\\"+FileName, mode ='r')as csvfile:
        csvreader = csv.reader(csvfile, delimiter=";")
        rows=[]
        for row in csvreader:
            rows.append(row)

    for i in range(len(rows)):
        CowNo.append(cowno)
        TagNo.append(tagno)
        # TimeStamp.append(rows[i][0])
        TimeStamp.append(datetime.strptime(rows[i][0],"%Y-%m-%dT%H:%M:%S.%f").timestamp())
        Ax.append(int(rows[i][1]))
        Ay.append(int(rows[i][2]))
        Az.append(int(rows[i][3]))
        Label.append(int(rows[i][4]))

    CowNo=list(CowNo)
    TagNo=list(TagNo)
    TimeStamp=list(TimeStamp)
    Label=list(Label)
    Ax=list(Ax)
    Ay=list(Ay)
    Az=list(Az)
    
    # Label=numpy.asarray(Label)
    # CowNo=numpy.asarray(CowNo)
    # TagNo=numpy.asarray(TagNo)
    return TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label

def LabeledDataSlicing(Ax, Ay, Az, Label, SliceN, Overlap, TagNo, CowNo, TimeStamp):
    AxSliced=list()
    AySliced=list()
    AzSliced=list()
    LabelSliced=list()

    TagNoSliced=list()
    CowNoSliced=list()
    TimeStampSliced=list()

    N=len(Ax)
    sliceX=numpy.zeros((SliceN, 1))
    sliceY=numpy.zeros((SliceN, 1))
    sliceZ=numpy.zeros((SliceN, 1))
    i=0
    CNprev=0
    while i<N:
        j=0
        L=Label[i]
        CN=CowNo[i]
        if CNprev!=CN:
            CNprev=CN
            print(CN)
        while i+j<N and Label[i+j]==L and CowNo[i+j]==CN and j<SliceN:
            sliceX[j]=Ax[i+j]
            sliceY[j]=Ay[i+j]
            sliceZ[j]=Az[i+j]
            j=j+1
        if j==SliceN:
            sliceX=AccNormalization(sliceX)
            sliceY=AccNormalization(sliceY)
            sliceZ=AccNormalization(sliceZ)
            AxSliced.append(sliceX.copy())
            AySliced.append(sliceY.copy())
            AzSliced.append(sliceZ.copy())
            LabelSliced.append(L)
            CowNoSliced.append(CN)
            TagNoSliced.append(TagNo[i])
            TimeStampSliced.append(TimeStamp[i])
        i=i+int(SliceN*Overlap)

    return AxSliced, AySliced, AzSliced, LabelSliced, TagNoSliced, CowNoSliced, TimeStampSliced

def AccelerationFiltering(Ax, Ay, Az):#WindowN,
    Draw=False#True#
    if Draw:
        plt.plot(TimeStamp,Ax, 'r.')
        plt.plot(TimeStamp,Ay, 'r.')
        plt.plot(TimeStamp,Az, 'r.')
    b = signal.firwin(511, cutoff = 0.1, fs = 25, window = "hamming", pass_zero="highpass")
    Ax = signal.lfilter(b, 1.0, Ax)
    Ay = signal.lfilter(b, 1.0, Ay)
    Az = signal.lfilter(b, 1.0, Az)
    # b, a = signal.butter(4, 25, 'low', analog=True)
    # b, a = signal.butter(4, 0.1)
    # Ax=signal.filtfilt(b,a,Ax)
    # Ay=signal.filtfilt(b,a,Ay)
    # Az=signal.filtfilt(b,a,Az)
    if Draw:
        plt.plot(TimeStamp,Ax,'bo')
        plt.plot(TimeStamp,Ay,'bo')
        plt.plot(TimeStamp,Az,'bo')
        plt.show()

    return Ax, Ay, Az

def AccelerationSamplingFitting(Ax, Ay, Az, TimeStamp, Label, TagNo, CowNo, RefFreq):#WindowN,
    Draw=False#True#
    if Draw:
        plt.plot(TimeStamp,Ax,'bo')
        plt.plot(TimeStamp,Ay,'bo')
        plt.plot(TimeStamp,Az,'bo')

    dt=1/RefFreq
    dterr=dt/2
    TimeStampNew=[]
    TimeStampNew.append(TimeStamp[0])
    q=[]
    q.append(0)
    j=0
    for i in range(len(TimeStamp)-1):
        if TimeStamp[i+1]-TimeStampNew[j]<5: #sec
            while TimeStampNew[j]+dt<TimeStamp[i+1]-dterr:
                TimeStampNew.append(TimeStampNew[j]+dt)
                j=j+1
                q.append(0)
        TimeStampNew.append(TimeStamp[i+1])
        j=j+1
        q.append(i+1)

    # AxNew,AyNew,AzNew=[],[],[]
    # for i in range(len(TimeStampNew)):
    #     if q[i]!=0:
    #         AxNew.append(Ax[q[i]])
    #         AyNew.append(Ay[q[i]])
    #         AzNew.append(Az[q[i]])
    #     else:
    #         AxNew.append(0)
    #         AyNew.append(0)
    #         AzNew.append(0)
    # Ax=AxNew
    # Ay=AyNew
    # Az=AzNew

    f=interpolate.interp1d(TimeStamp,Ax,kind='cubic')
    Ax=f(TimeStampNew)
    f=interpolate.interp1d(TimeStamp,Ay,kind='cubic')
    Ay=f(TimeStampNew)
    f=interpolate.interp1d(TimeStamp,Az,kind='cubic')
    Az=f(TimeStampNew)
    f=interpolate.interp1d(TimeStamp,TagNo,kind='nearest')
    TagNo=f(TimeStampNew)
    f=interpolate.interp1d(TimeStamp,CowNo,kind='nearest')
    CowNo=f(TimeStampNew)
    f=interpolate.interp1d(TimeStamp,Label,kind='nearest')
    Label=f(TimeStampNew)

    if Draw:
        plt.plot(TimeStampNew,Ax, 'g*')
        plt.plot(TimeStampNew,Ay, 'g*')
        plt.plot(TimeStampNew,Az, 'g*')

        # # plt.figure()
        # # plt.plot(TimeStampNew,TagNo, 'k.')
        # # plt.plot(TimeStampNew,CowNo, 'k*')

        plt.show()

    return Ax, Ay, Az, TimeStampNew, Label, TagNo, CowNo

def AccNormalization(A):
    return A/numpy.std(A)

def DataBalance(Ax, Ay, Az, Label, CowNo):#TimeStamp
    N=len(Label)
    m=max(Label)

    (unique, LabelCounts) = numpy.unique(Label, return_counts=True)
    print("Total class distribution:"+str(LabelCounts))
    LabelN=len(LabelCounts)

    CowNoList=list(set(CowNo))
    CowNo=numpy.array(CowNo)
    Ax=numpy.array(Ax)
    Ay=numpy.array(Ay)
    Az=numpy.array(Az)
    Label=numpy.array(Label)
    # TimeStamp=numpy.array(TimeStamp)
    AxBalanced=list()
    AyBalanced=list()
    AzBalanced=list()
    LabelBalanced=list()
    CowNoBalanced=list()

    for Cow_i in range(len(CowNoList)):
        q=numpy.argwhere(CowNo==CowNoList[Cow_i])
        AxCowNo_i=Ax[q]
        AyCowNo_i=Ay[q]
        AzCowNo_i=Az[q]
        LabelCowNo_i=Label[q]

        for i in range(len(LabelCowNo_i)):
            AxBalanced.append(numpy.squeeze(AxCowNo_i[i]))
            AyBalanced.append(numpy.squeeze(AyCowNo_i[i]))
            AzBalanced.append(numpy.squeeze(AzCowNo_i[i]))
            LabelBalanced.append(LabelCowNo_i[i][0])
            CowNoBalanced.append(CowNoList[Cow_i])

        (unique, LabelCounts) = numpy.unique(LabelCowNo_i, return_counts=True)
        print("Cow "+str(CowNoList[Cow_i])+", classes "+str(LabelCounts))
        LabelCountsMax=numpy.max(LabelCounts)
        for Label_i in range(LabelN):
            e=numpy.argwhere(LabelCowNo_i==(Label_i+1))
            AxCowNo_iLabel_i=AxCowNo_i[e[:,0]]
            AyCowNo_iLabel_i=AyCowNo_i[e[:,0]]
            AzCowNo_iLabel_i=AzCowNo_i[e[:,0]]
            for j in range(LabelCountsMax-LabelCounts[Label_i]):
                r=numpy.random.randint(LabelCounts[Label_i])
                ax0=numpy.squeeze(AxCowNo_iLabel_i[r])
                ay0=numpy.squeeze(AyCowNo_iLabel_i[r])
                az0=numpy.squeeze(AzCowNo_iLabel_i[r])
                RotA=math.pi*numpy.random.rand()
                s=math.sin(RotA)
                c=math.cos(RotA)
                ax=ax0
                ay=ay0*c-az0*s
                az=ay0*s+az0*c
                AxBalanced.append(ax)
                AyBalanced.append(ay)
                AzBalanced.append(az)
                LabelBalanced.append(Label_i+1)
                CowNoBalanced.append(CowNoList[Cow_i])

        # # TimeStampCowNo_i = numpy.array(TimeStamp[q], dtype='datetime64[D]')
        # # TimeStampListCowNo_i=numpy.unique(TimeStampCowNo_i)

        # a=numpy.array(numpy.squeeze(TimeStamp[q]/(60*60*24)))
        # print(a)

        # TimeStampCowNo_i = numpy.array(numpy.floor(numpy.squeeze(TimeStamp[q])/(60*60*24)))
        # TimeStampListCowNo_i=numpy.unique(TimeStampCowNo_i)
        # for Date_i in range(len(TimeStampListCowNo_i)):
        #     w=numpy.argwhere(TimeStampCowNo_i==TimeStampListCowNo_i[Date_i])
        #     AxCowNo_iDate_i=AxCowNo_i[w[:,0]]
        #     AyCowNo_iDate_i=AyCowNo_i[w[:,0]]
        #     AzCowNo_iDate_i=AzCowNo_i[w[:,0]]
        #     LabelCowNo_iDate_i=LabelCowNo_i[w[:,0]]
        #     for i in range(len(LabelCowNo_iDate_i)):
        #         AxBalanced.append(numpy.squeeze(AxCowNo_iDate_i[i]))
        #         AyBalanced.append(numpy.squeeze(AyCowNo_iDate_i[i]))
        #         AzBalanced.append(numpy.squeeze(AzCowNo_iDate_i[i]))
        #         LabelBalanced.append(LabelCowNo_iDate_i[i][0])
        #         CowNoBalanced.append(CowNoList[Cow_i])

        #     (unique, LabelCounts) = numpy.unique(LabelCowNo_iDate_i, return_counts=True)
        #     print("Cow "+str(CowNoList[Cow_i])+", date "+str(TimeStampListCowNo_i[Date_i])+", classes "+str(LabelCounts))
        #     LabelCountsMax=numpy.max(LabelCounts)
        #     for Label_i in range(3):
        #         e=numpy.argwhere(LabelCowNo_iDate_i==(Label_i+1))
        #         AxCowNo_iDate_iLabel_i=AxCowNo_iDate_i[e[:,0]]
        #         AyCowNo_iDate_iLabel_i=AyCowNo_iDate_i[e[:,0]]
        #         AzCowNo_iDate_iLabel_i=AzCowNo_iDate_i[e[:,0]]
        #         for j in range(LabelCountsMax-LabelCounts[Label_i]):
        #             r=numpy.random.randint(LabelCounts[Label_i])
        #             # ax0=AxCowNo_iDate_iLabel_i[r]
        #             # ay0=AyCowNo_iDate_iLabel_i[r]
        #             # az0=AzCowNo_iDate_iLabel_i[r]
        #             ax0=numpy.squeeze(AxCowNo_iDate_iLabel_i[r])
        #             ay0=numpy.squeeze(AyCowNo_iDate_iLabel_i[r])
        #             az0=numpy.squeeze(AzCowNo_iDate_iLabel_i[r])
        #             RotA=math.pi*numpy.random.rand()
        #             s=math.sin(RotA)
        #             c=math.cos(RotA)
        #             ax=ax0
        #             ay=ay0*c-az0*s
        #             az=ay0*s+az0*c
        #             AxBalanced.append(ax)
        #             AyBalanced.append(ay)
        #             AzBalanced.append(az)
        #             LabelBalanced.append(Label_i+1)
        #             CowNoBalanced.append(CowNoList[Cow_i])

    (unique, LabelCounts) = numpy.unique(LabelBalanced, return_counts=True)
    print("Balanced class distribution: "+str(LabelCounts))

    return AxBalanced, AyBalanced, AzBalanced, LabelBalanced, CowNoBalanced

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


def DataSlicing(Ax, Ay, Az, Timestamp, n):
    AxSliced=list()
    AySliced=list()
    AzSliced=list()
    TimestampSliced=list()
    N=len(Ax)
    sliceX=numpy.zeros((n, 1))
    sliceY=numpy.zeros((n, 1))
    sliceZ=numpy.zeros((n, 1))
    i=0
    while i+n<N:
        for j in range(n):
            sliceX[j]=Ax[i+j]
            sliceY[j]=Ay[i+j]
            sliceZ[j]=Az[i+j]
        sliceX=AccNormalization(sliceX)
        sliceY=AccNormalization(sliceY)
        sliceZ=AccNormalization(sliceZ)
        AxSliced.append(sliceX.copy())
        AySliced.append(sliceY.copy())
        AzSliced.append(sliceZ.copy())
        TimestampSliced.append(Timestamp[i])#+n//2
        i=i+j

    return AxSliced, AySliced, AzSliced, TimestampSliced

def MaxInd(a):
    m=a[0]
    k=0
    for i in range(1,len(a)):
        if m<a[i]:
            k=i
            m=a[i]
    return m,k

def AccAugmentation(Ax0, Ay0, Az0, Label0, CowNo0, RotationXN):
    Ax=list()
    Ay=list()
    Az=list()
    Label=list()
    CowNo=list()
    if RotationXN==0:
        RotationXN=1

    dRotationX=math.pi/(RotationXN)
    for i in range(len(Ax0)):
        ax0=Ax0[i]
        ay0=Ay0[i]
        az0=Az0[i]

        for Rotation_i in range(RotationXN*0+1):
            RotA=2*math.pi*numpy.random.rand()
            s=math.sin(RotA)
            c=math.cos(RotA)

            # s=math.sin(Rotation_i*dRotationX)
            # c=math.cos(Rotation_i*dRotationX)
            ax=ax0
            ay=ay0*c-az0*s
            az=ay0*s+az0*c
            Ax.append(ax)
            Ay.append(ay)
            Az.append(az)
            Label.append(Label0[i])
            CowNo.append(CowNo0[i])

    (unique, LabelCounts) = numpy.unique(Label, return_counts=True)
    print("Augmented class distribution: "+str(LabelCounts))

    return Ax, Ay, Az, Label, CowNo

from sklearn import metrics
def PresentPerformance(LabelReference,LabelPredicted,Classes,ClassNames):
    N=len(LabelPredicted)
    ClassN=len(Classes)
    Ttotal = 0
    TP=numpy.zeros((ClassN, 1))
    TN=numpy.zeros((ClassN, 1))
    FP=numpy.zeros((ClassN, 1))
    FN=numpy.zeros((ClassN, 1))
    for i in range(N):
        if LabelPredicted[i]==LabelReference[i]:
            Ttotal=Ttotal+1

        for j in range(ClassN):
            if LabelReference[i]==Classes[j]:
                if LabelPredicted[i]==Classes[j]:
                    TP[j]=TP[j]+1
                else:
                    FN[j]=FN[j]+1
            else:
                if LabelPredicted[i]==Classes[j]:
                    FP[j]=FP[j]+1
                else:
                    TN[j]=TN[j]+1

    PerfVect=numpy.zeros((ClassN, 3))
    print()
    TotAcc=round(Ttotal/N*100,2)
    print('Total accuracy =',TotAcc,'%, N =',N)
    for j in range(ClassN):
        sens=numpy.squeeze(TP[j]/(TP[j]+FP[j]))
        spec=numpy.squeeze(TN[j]/(TN[j]+FP[j]))
        F1=2*sens*spec/(sens+spec)
        print(ClassNames[j],' sens =',numpy.round(sens*100,2),'%, spec =',numpy.round(spec*100,2),'%, F1 =',numpy.round(F1*100,2),'%, N =',numpy.squeeze(int(TP[j]+TN[j])))
        PerfVect[j]=[sens, spec, F1]

    ConfusionMatr=numpy.zeros((3, 3))
    for i in range(N):
        ConfusionMatr[int(LabelPredicted[i])-1,int(LabelReference[i])-1]+=1

    print()
    print(numpy.squeeze(ClassNames))
    # print(numpy.squeeze(ConfusionMatr.astype(int)))

    print(metrics.confusion_matrix(LabelReference,LabelPredicted))
    print(metrics.classification_report(LabelReference,LabelPredicted, digits=3))

    return PerfVect, ConfusionMatr, TotAcc

from scipy.stats import skew, kurtosis
def AccelerationFeatures(Ax,Ay,Az,Label):
    Ax_Mean=numpy.mean(Ax, axis=1)
    Ay_Mean=numpy.mean(Ay, axis=1)
    Az_Mean=numpy.mean(Az, axis=1)
    Ax_STD=numpy.std(Ax, axis=1)
    Ay_STD=numpy.std(Ay, axis=1)
    Az_STD=numpy.std(Az, axis=1)
    Ax_Skew=skew(Ax)
    Ay_Skew=skew(Ay)
    Az_Skew=skew(Az)
    Ax_Kurt=kurtosis(Ax)
    Ay_Kurt=kurtosis(Ay)
    Az_Kurt=kurtosis(Az)
    Features=numpy.dstack((Ax_Mean,Ay_Mean,Az_Mean))
    Features=numpy.dstack((Features,Ax_STD,Ay_STD,Az_STD))
    # Features=numpy.dstack((Features,Ax_Skew,Ay_Skew,Az_Skew))
    # Features=numpy.dstack((Features,Ax_Kurt,Ay_Kurt,Az_Kurt))

    Features=numpy.squeeze(Features)
    Label=numpy.squeeze(Label)
    return Features,Label



    
# def ReadLabeledData(DataFolder,FileName):#
#     with open(DataFolder+"\\"+FileName, mode ='r')as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=";")
#         rows=[]
#         for row in csvreader:
#             rows.append(row)
#     Ax=[]
#     Ay=[]
#     Az=[]
#     Label=[]
#     TimeStamp=[]
#     CowNo=[]
#     TagNo=[]
#     for i in range(len(rows)):
#         CowNo.append(int(rows[i][0]))
#         TagNo.append(int(rows[i][1]))
#         TimeStamp.append(rows[i][2])
#         Ax.append(int(rows[i][3]))
#         Ay.append(int(rows[i][4]))
#         Az.append(int(rows[i][5]))
#         Label.append(int(rows[i][6]))
#     Ax=AccNormalization(Ax)
#     Ay=AccNormalization(Ay)
#     Az=AccNormalization(Ay)
#     Label=numpy.asarray(Label)
#     return TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label       

# def ReadLabeledCSV(DataFolder,FileName):#
#     rows = [] 
#     print(DataFolder+"\\"+FileName)
#     with open(DataFolder+"\\"+FileName, mode ='r')as csvfile: 
#         csvreader = csv.reader(csvfile, delimiter=";")
#         for row in csvreader:
#             rows.append(row)
#     Ax=[]
#     Ay=[]
#     Az=[]
#     Label=[]
#     # TimeStamp=[]
#     for i in range(len(rows)):
#         # TimeStamp.append(float(rows[i][0]))
#         # TimeStamp.append(rows[i][0])
#         Ax.append(float(rows[i][1]))
#         Ay.append(float(rows[i][2]))
#         Az.append(float(rows[i][3]))
#         Label.append(int(rows[i][4]))
#     return Ax, Ay, Az, Label#, TimeStamp

# def ReadRuuviCSV(DataFolder,FileName):
#     rows = [] 
#     print(DataFolder+"\\"+FileName)
#     with open(DataFolder+"\\"+FileName, mode ='r')as csvfile: 
#         csvreader = csv.reader(csvfile, delimiter=";")
#         for row in csvreader:
#             rows.append(row)
#     Ax=[]
#     Ay=[]
#     Az=[]
#     Timestamp=[]
#     for i in range(len(rows)):
#         Ax.append(float(rows[i][1]))
#         Ay.append(float(rows[i][2]))
#         Az.append(float(rows[i][3]))
#         Timestamp.append(rows[i][0])
#     return Ax, Ay, Az, Timestamp
