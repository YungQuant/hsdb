import os
import math
import sklearn as sk
import requests, pandas as pan
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.metrics import mean_squared_error
import scipy.stats as sp
import keras
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU, PReLU
from keras.layers import LSTM, Dropout
from keras import backend as K
import numpy as np
import time, datetime


def writeDirectionalDataset(X, Y, path="../..HSDB_unnamedDataset.txt"):
    path = path + ".txt"

    if os.path.isfile(path) == True:
        print(f'Found {path}, not reproducing')
        return

    fileP = open(path, "w")

    for i in range(len(X)):
        try:
            strX, strY = "", ""
            for k in range(len(X[i])):
                strX += str(X[i][k])
                strX += ","

            strY = str(Y[i])

            fileP.write(strX[:-1] + "\n")
            fileP.write(strY + "\n")

        except Exception as e:
            print(e)

    fileP.close()

def writeDirectionalClassDataset(X, Y, path="../..HSDB_unnamedDataset.txt"):
    path = path + ".txt"

    if os.path.isfile(path) == True:
        print(f'Found {path}, not reproducing')
        return

    fileP = open(path, "w")

    for i in range(len(X)):
        try:
            strX, strY = "", ""
            for k in range(len(X[i])):
                strX += str(X[i][k])
                strX += ","

            for j in range(len(Y[i])):
                strY += str(Y[i][j])
                strY += ","

            fileP.write(strX[:-1] + "\n")
            fileP.write(strY[:-1] + "\n")

        except Exception as e:
            print(e)

    fileP.close()

def readDirectionalDataset(path="../..HSDB_unnamedDataset.txt"):
    path = "../../datasets/" + path
    if os.path.isfile(path) == False:
        print(f'{path}, not found')
    fileP = open(path, "r")
    lines = fileP.readlines()

    X, Y = [], []

    i = 0
    while i < len(lines)-1:
        linex = lines[i].split(",")
        liney = lines[i+1].split(",")

        X.append([float(l) for l in linex[:-1]])
        Y.append([float(l) for l in liney[:-1]])

        i += 2

    return np.array(X), np.array(Y)

def readDirectionalClassDataset(path="../..HSDB_unnamedDataset.txt", old=False):
    path = "../../datasets/" + path
    if os.path.isfile(path) == False:
        print(f'{path}, not found')
    fileP = open(path, "r")
    lines = fileP.readlines()

    X, Y = [], []

    i = 0
    while i < len(lines)-1:
        if old == True:
            linex = lines[i].split(",")
            # [0 1 0]
            liney = str(lines[i+1])

            # print(f'linex: {linex} \n')
            # print(f'liney: {liney} \n')
            # [print(f'{t}' for t in liney)]

            X.append([float(l) for l in linex[:-1]])
            Y.append([float(liney[1]), float(liney[3]), float(liney[5])])
        else:
            linex = lines[i].split(",")
            liney = lines[i+1].split(",")

            X.append([float(l) for l in linex])
            Y.append([float(l) for l in liney])

        i += 2

    return np.array(X), np.array(Y)

def readAndClassifyDataset(path="../..HSDB_unnamedDataset.txt"):
    path = "../../datasets/" + path
    if os.path.isfile(path) == False:
        print(f'{path}, not found')
    fileP = open(path, "r")
    lines = fileP.readlines()

    X, Y = [], []

    i = 0
    while i < len(lines)-1:
        linex = lines[i].split(",")
        liney = lines[i+1].split(",")

        X.append([float(l) for l in linex[:-1]])
        c = float(liney[0])
        if c > 0:
                Y.append([0, 0, 1])
        elif c == 0:
            Y.append([0, 1, 0])
        elif c < 0:
            Y.append([1, 0, 0])

        i += 2

    return np.array(X), np.array(Y)

def forza(currency="BMEX_XBTUSD2_100kus", depth=30, p=1, s=1, scale=10000, volOnly=False):
    # path = f'../../HSDB_{currency}.txt'
    path = f'../../HSDB-{currency}.txt'
    # /home/yungquant/HSDB-BMEX_XBTUSD2_100kus.txt
    data, datum = [], []
    if os.path.isfile(path) == False:
        print(f'could not source {path} data')
    else:
        fileP = open(path, "r")
        lines = fileP.readlines()
        lines = lines[:int(np.floor(p * len(lines)))]

        i, k = 0, 0
        while i < len(lines)-3:
            # bidP = lines[i].split(",")[:depth]
            # bidV = lines[i+1].split(",")[:depth]
            # askP = list(reversed(lines[i+2].split(",")))[:depth]
            # askV = list(reversed(lines[i+3].split(",")))[:depth]

            # for k in range(depth):
            #     datum.append(float(bidP[k]))
            # for k in range(depth):
            #     datum.append(float(bidV[k]))
            # for k in range(depth):
            #     datum.append(float(askP[k]))
            # for k in range(depth):
            #     datum.append(float(askV[k]))
            if k % s == 0:
                if volOnly == False:
                    for k in range(depth):
                        datum.append(float(list(reversed(lines[i].split(",")[:depth]))[k]))
                for k in range(depth):
                    datum.append(float(list(reversed(lines[i+1].split(",")[:depth]))[k])/scale)
                if volOnly == False:
                    for k in range(depth):
                        datum.append(float(list(reversed(lines[i+2].split(",")))[:depth][k]))
                for k in range(depth):
                    datum.append(float(list(reversed(lines[i+3].split(",")))[:depth][k])/scale)

                if volOnly == False and datum[0] > 1000  and datum[0] < 500000 and (datum[depth-1] < datum[depth*3-1]):
                    data.append(datum)
                else:
                    i-=3
                datum = []

            i+=4
            k+=1

    return data

def create_forzaMidpoint_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append(np.mean([dataset[i+1][0], dataset[i+1][60]]))
    return np.array(dataX), np.array(dataY)

def create_forzaSpread_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append(dataset[i+1][60] - dataset[i+1][0])
    return np.array(dataX), np.array(dataY)

def create_forzaDirection_dataset(dataset, distance=1, depth=30):
    dataX, dataY = [], []
    for i in range(len(dataset)-distance):
        dataX.append(dataset[i])
        try:
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]]))
        except:
            print("create_forzaDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][depth*2], dataset[i+distance][0], dataset[i][depth*2], dataset[i][0])

    return np.array(dataX), np.array(dataY)


def create_forzaSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100, vO=False):
    dataX, dataY = [], []
    if vO == True:
        dims = depth
    else:
        dims = depth*2

    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            for j in k:
                datum1.append(j)
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][dims]]) - np.mean([dataset[i][0], dataset[i][dims]]))
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][dims], dataset[i+distance][0], dataset[i][dims], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaSequentialClassDirection_dataset(dataset, distance=1, depth=30, lookback=100, vO=False):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            if vO == False:
                for j in k:
                    datum1.append(j)
            else:
                for j in k[depth:depth*2]:
                    datum1.append(j)
                for jk in k[depth*3:depth*4]:
                    datum1.append(jk)
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            c = (np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]]))
       
            
            if c > 0:
                dataY.append([0, 0, 1])
            elif c == 0:
                dataY.append([0, 1, 0])
            elif c < 0:
                dataY.append([1, 0, 0])
            
        except:
            print("create_forzaSequentialClassDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][depth*2], dataset[i+distance][0], dataset[i][depth*2], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaFortuneTeller_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append([dataset[i+1][0], dataset[i+1][1], dataset[i+1][2], dataset[i+1][3], dataset[i+1][60], dataset[i+1][61], dataset[i+1][62], dataset[i+1][63]])
    return np.array(dataX), np.array(dataY)
    
def volatilityAdaptiveClassFilter(X, Y, attention=1000):
    x, y = [], []
    for i in range(attention, len(Y)):
        for k in Y[i-attention:i]:
            max = max(k)
            maxIx = list(k).index(max)
            min = min(k)
            minIx = list(k).index(min)
            if maxIx == 2 or minIx == 0:
                x.append(X[i])
                y.append(Y[i])
                break
    return np.array(x), np.array(y)

# maxP = max(pY)
# maxPIx = list(pY).index(maxP)

def cosl(e):
    if e <= 10:
        return (math.tanh(((math.cos(e+1) * 3.14)+1)/10)/(math.log(e+1)+1))+ 0.0000001
    elif e <= 20:
        return (math.tanh(((math.cos(e+1) * 3.14)+1)/10)/(math.log(e+1)+1)) * 2 + 0.0000001
    elif e <= 30:
        return (math.tanh(((math.cos(e+1) * 3.14)+1)/10)/(math.log(e+1)+1)) * 3 + 0.0000001
    elif e <= 50:
        return (math.tanh(((math.cos(e+1) * 3.14)+1)/10)/(math.log(e+1)+1)) * 4 + 0.0000001
    elif e <= 80:
        return (math.tanh(((math.cos(e+1) * 3.14)+1)/10)/(math.log(e+1)+1)) * 5 + 0.0000001


class hsdbSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

def plot(y):
    plt.plot(np.arange(len(y)), y)
    plt.show()


# TO-DO: transformer & reinforcement
# TO-DO: train inline on file stream
# TO-DO: paralellize or c-ify preprocessing
# TO-DO: tensorboard (callback)

# path = input("Enter data path:")

timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
currency_pairs, currencies = ["XBTUSD", "ETHUSD", "XRPU18", "LTCU18", "BCHU18"], ["BTCUSD", "ADABTC", "ETHUSD", "LTCBTC", "XRPBTC"]
Dfiles = ["XBTUSD02", "BMEX_XBTUSD2_100kus", "BMEX_XBTUSD3_100kus", "BMEX_XBTUSD3_10kus", "BMEX_XBTUSD667_100kus", 
"HSDBdirectionalLSTM0-Din20-dist3-lookback30-perc0.99-sparcity10-datasetBMEX_XBTUSD2_100kus-time2019-04-20T02:40:32.644265UTC__2019-04-20T02:45:14.197736UTC.txt", 
"HSDBdirectionalClassLSTM0-MidPeak-Din20-dist10-lookback20-perc0.99-sparcity10-scale100000-vOTrue-datasetBMEX_XBTUSD3_100kus.txt",
 "BMEX_XBTUSD5_10kus"]
path = Dfiles[4]
errs, Ps, aPs, aTestYs, aTrainYs, passes, fails = [], [], [], [], [], 0, 0
Din = 10; dist = 10; perc = 0.95; c = 1.5; b = 32; nb_epoch = 50; l = 20; opt = "Adadelta"; s = 10; scale = 10000000; vO = True
if vO == True:
    dims = Din*2*l
else:
    dims = Din*4*l
X, Y = create_forzaSequentialClassDirection_dataset(forza(path, Din, perc, s, scale, volOnly=False), dist, Din, l, vO=vO)
# X, Y = readDirectionalClassDataset(path, old=True)
print("X0: ", X[0], " Y0 ", Y[0])
# plot(X[0])
writeDirectionalClassDataset(X, Y, f'../../datasets/HSDBdirectionalClassLSTM0-MidPeak-Din{Din}-dist{dist}-lookback{l}-perc{perc}-sparcity{s}-scale{scale}-vO{vO}-dataset{path}')
#print("\nshape(X):", X.shape)

trainX, trainY, testX, testY = X[:int(np.floor(len(X)/c))], Y[:int(np.floor(len(X)/c))], X[int(np.floor(len(X)/c)):], Y[int(np.floor(len(X)/c)):]
# print(testX[0], testY[0])

# scaler = MinMaxScaler(feature_range=(-10, 10))
# trainX = scaler.fit_transform(trainX)
# testX = scaler.fit_transform(testX)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print("scaled training x[-1], y[-1]", trainX[-1], trainY[-1])
print("trainX shape", trainX.shape, "testX shape", testX.shape)
lrs = LearningRateScheduler(cosl)
stop = EarlyStopping(patience=20)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=10, min_lr=0.0000000001)
# saver = keras.callbacks.ModelCheckpoint(f'../models/hsdbDirectionalClassLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}_time{timeStr}.h5',
                                        # monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# logF = f'../logs/hsdbDirectionalClassLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}_time{timeStr}.txt'
# opt = keras.optimizers.Adam(lr=0, epsilon=0.00000001, decay=0.0001, amsgrad=False)

K.tensorflow_backend._get_available_gpus()
model = Sequential()
model.add(Dense(dims, input_shape=(1, dims)))
model.add(LeakyReLU(alpha=0.111))
# model.add(Dropout(0.5))
# model.add(Dense(Din*4*l, input_shape=(1, Din*4*l)))
# model.add(Dense(Din*4*l, activation='relu'))
model.add(LSTM(dims, return_sequences=True))
model.add(LeakyReLU(alpha=0.333))
# model.add(Dropout(0.5))
model.add(LSTM(dims, return_sequences=False))
model.add(LeakyReLU(alpha=0.111))
model.add(Dense(dims, activation='tanh'))
# model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
model.fit_generator(hsdbSequence(trainX, trainY, b), steps_per_epoch=(len(trainX) / b),
                                          epochs=nb_epoch,
                                          verbose=2,
                                          validation_data=hsdbSequence(testX, testY, b),
                                          validation_steps=(len(testX) / b),
                                          use_multiprocessing=False,
                                          workers=8,
                                          max_queue_size=2,
                                          callbacks=[stop])
# model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=5, verbose=2, callbacks=[reduce_lr, saver])
# TO-DO 50%: write formatted datasets to file, read from files on training to save reformatting time
# FOR UNIDIMENTIONAL PREDICTIONS VVV

for i in range(len(testX)):
    sTXi = np.reshape(testX[i], (testX[i].shape[0], 1, testX[i].shape[1]))
    pY, rY = model.predict(sTXi)[0], testY[i]
    # print("sTXi:", sTXi)
    # print("pY:", pY, "rY:", rY)
    maxP = max(pY)
    maxPIx = list(pY).index(maxP)
    maxY = max(rY)
    maxYIx = list(rY).index(maxY)
    if maxPIx == maxYIx:
        passes += 1
    else:
        fails += 1
    # TO-DO: INVERSE ACCURACY
    Ps.append(pY)
    aPs.append(maxPIx)
    aTestYs.append(maxYIx)
    errs.append(abs(maxPIx - maxYIx)/3 * 100)
    # print("sTXi:", sTXi)
    # print("pY:", pY, "rY:", rY, "err %:", errs[-1])

print("\n\nAggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY), 
    "Mean Perc. Error:", np.mean(errs))

for y in trainY:
    maxTY = max(y)
    maxTYIx = list(y).index(maxTY)
    aTrainYs.append(maxTYIx)

print("\nPs Describe:", sp.describe(aPs), "Ps Median:", list(sorted(aPs))[int(np.floor(len(aPs)/2))], "\ntrainY Describe:", sp.describe(aTrainYs), "\ntestY Describe:", sp.describe(aTestYs))

print("\n\n SAVE MODEL?")
resp = input()
if resp in ["y", "yes", "Y", "YES", "save", "SAVE"]:
    model.save(f'../models/hsdbDirectionalLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}-mpe{np.mean(errs)}-aba{passes / len(testY)}-_{timeStr}.h5')
