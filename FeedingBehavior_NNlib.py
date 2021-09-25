import csv
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.utils import to_categorical
import tensorflow
import fnmatch, os
import math
import datetime
from scipy import signal
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
    elif ModelName=="NN4":
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=32, strides=1, activation='relu', input_shape = (n_timesteps, n_features)))
        model.add(Conv1D(filters=64, kernel_size=32, strides=2, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=32, strides=2, activation='relu'))
        model.add(Conv1D(filters=512, kernel_size=1, strides=1, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax', dtype='float32'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy", "categorical_accuracy"])
    else:
        print()
    return model

def ReadLabeledDataFiles(DataFolder,Mask):
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
            with open(DataFolder+"\\"+FileName, mode ='r')as csvfile:
                csvreader = csv.reader(csvfile, delimiter=";")
                rows=[]
                for row in csvreader:
                    rows.append(row)

            for i in range(len(rows)):
                CowNo.append(int(rows[i][0]))
                TagNo.append(int(rows[i][1]))
                TimeStamp.append(rows[i][2])
                Ax.append(int(rows[i][3]))
                Ay.append(int(rows[i][4]))
                Az.append(int(rows[i][5]))
                Label.append(int(rows[i][6]))
    
    Label=numpy.asarray(Label)
    return TagNo, CowNo, TimeStamp, Ax, Ay, Az, Label

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

# def AccNormalization(A):
#     mu = numpy.mean(A)
#     sigma = numpy.std(A)
#     return (A - mu)/sigma
    
def AccNormalization(A):
    b = signal.firwin(511, cutoff = 0.1, fs = 25, window = "hamming", pass_zero="highpass")
    Af = signal.lfilter(b, 1.0, A)

    # plt.scatter(range(len(A)), A)
    # plt.scatter(range(len(A)), Af)
    # plt.show()

    return Af/numpy.std(Af)

def DataBalance(Ax, Ay, Az, Label, CowNo, TimeStamp):
    N=len(Label)
    m=max(Label)

    (unique, LabelCounts) = numpy.unique(Label, return_counts=True)
    print("Total class distribution:"+str(LabelCounts))

    CowNoList=list(set(CowNo))
    CowNo=numpy.array(CowNo)
    Ax=numpy.array(Ax)
    Ay=numpy.array(Ay)
    Az=numpy.array(Az)
    Label=numpy.array(Label)
    TimeStamp=numpy.array(TimeStamp)
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
        TimeStampCowNo_i = numpy.array(TimeStamp[q], dtype='datetime64[D]')
        TimeStampListCowNo_i=numpy.unique(TimeStampCowNo_i)

        for Date_i in range(len(TimeStampListCowNo_i)):
            w=numpy.argwhere(TimeStampCowNo_i==TimeStampListCowNo_i[Date_i])
            AxCowNo_iDate_i=AxCowNo_i[w[:,0]]
            AyCowNo_iDate_i=AyCowNo_i[w[:,0]]
            AzCowNo_iDate_i=AzCowNo_i[w[:,0]]
            LabelCowNo_iDate_i=LabelCowNo_i[w[:,0]]
            for i in range(len(LabelCowNo_iDate_i)):
                AxBalanced.append(numpy.squeeze(AxCowNo_iDate_i[i]))
                AyBalanced.append(numpy.squeeze(AyCowNo_iDate_i[i]))
                AzBalanced.append(numpy.squeeze(AzCowNo_iDate_i[i]))
                LabelBalanced.append(LabelCowNo_iDate_i[i][0])
                CowNoBalanced.append(CowNoList[Cow_i])

            (unique, LabelCounts) = numpy.unique(LabelCowNo_iDate_i, return_counts=True)
            print("Cow "+str(CowNoList[Cow_i])+", date "+str(TimeStampListCowNo_i[Date_i])+", classes "+str(LabelCounts))
            LabelCountsMax=numpy.max(LabelCounts)
            for Label_i in range(3):
                e=numpy.argwhere(LabelCowNo_iDate_i==(Label_i+1))
                AxCowNo_iDate_iLabel_i=AxCowNo_iDate_i[e[:,0]]
                AyCowNo_iDate_iLabel_i=AyCowNo_iDate_i[e[:,0]]
                AzCowNo_iDate_iLabel_i=AzCowNo_iDate_i[e[:,0]]
                for j in range(LabelCountsMax-LabelCounts[Label_i]):
                    r=numpy.random.randint(LabelCounts[Label_i])
                    # ax0=AxCowNo_iDate_iLabel_i[r]
                    # ay0=AyCowNo_iDate_iLabel_i[r]
                    # az0=AzCowNo_iDate_iLabel_i[r]
                    ax0=numpy.squeeze(AxCowNo_iDate_iLabel_i[r])
                    ay0=numpy.squeeze(AyCowNo_iDate_iLabel_i[r])
                    az0=numpy.squeeze(AzCowNo_iDate_iLabel_i[r])
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

    (unique, LabelCounts) = numpy.unique(LabelBalanced, return_counts=True)
    print("Balanced class distribution: "+str(LabelCounts))

    return AxBalanced, AyBalanced, AzBalanced, LabelBalanced, CowNoBalanced

def ReadRuuviCSV(DataFolder,FileName):
    rows = [] 
    print(DataFolder+"\\"+FileName)
    with open(DataFolder+"\\"+FileName, mode ='r')as csvfile: 
        csvreader = csv.reader(csvfile, delimiter=";")
        for row in csvreader:
            rows.append(row)
    Ax=[]
    Ay=[]
    Az=[]
    Timestamp=[]
    for i in range(len(rows)):
        Ax.append(float(rows[i][1]))
        Ay.append(float(rows[i][2]))
        Az.append(float(rows[i][3]))
        Timestamp.append(rows[i][0])
    return Ax, Ay, Az, Timestamp

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

        for Rotation_i in range(RotationXN):
            s=math.sin(Rotation_i*dRotationX)
            c=math.cos(Rotation_i*dRotationX)
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
    print('Total accuracy =',round(Ttotal/N*100,2),'%, N =',N)
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
    print(numpy.squeeze(ConfusionMatr.astype(int)))

    return PerfVect, ConfusionMatr
