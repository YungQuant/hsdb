import os
import sklearn as sk
import requests, pandas as pan
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.metrics import mean_squared_error
import scipy.stats as sp
import keras
from keras.callbacks import ReduceLROnPlateau
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

            fileP.write(strX + "\n")
            fileP.write(strY + "\n")

        except Exception as e:
            print(e)

    fileP.close()

def readDataset(path="../..HSDB_unnamedDataset.txt"):
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
    

def forza(currency="BMEX_XBTUSD2_100kus", depth=30, p=1, s=1, volOnly=False):
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
                        datum.append(float(lines[i].split(",")[:depth][k]))
                for k in range(depth):
                    datum.append(float(lines[i+1].split(",")[:depth][k]))
                if volOnly == False:
                    for k in range(depth):
                        datum.append(float(list(reversed(lines[i+2].split(",")))[:depth][k]))
                for k in range(depth):
                    datum.append(float(list(reversed(lines[i+3].split(",")))[:depth][k]))

                data.append(datum)
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

def create_forzaSequentialClassDirection_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            for j in k:
                datum1.append(j)
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]])*2)
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
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

def create_forzaSparseSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100, sparcity=10):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
        if i % sparcity == 0:
            datum1 = []
            for k in dataset[i-lookback:i]:
                for j in k:
                    datum1.append(j)
            try:
                dataX.append(datum1)
                # dataX.append([j for j in k for k in dataset[i-lookback:i]])
                dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]]))
            except:
                print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
                print("dataset[i+distance]: ", dataset[i+distance])
                # print(dataset[i+distance], "\n", dataset[i])
                print(dataset[i+distance][depth*2], dataset[i+distance][0], dataset[i][depth*2], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaNerfedSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
        if np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]]) != 0:
            datum1 = []
            for k in dataset[i-lookback:i]:
                for j in k:
                    datum1.append(j)
            try:
                dataX.append(datum1)
                dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]]))
            except:
                print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
                print("dataset[i+distance]: ", dataset[i+distance])
                # print(dataset[i+distance], "\n", dataset[i])
                print(dataset[i+distance][depth*2], dataset[i+distance][0], dataset[i][depth*2], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaPartiallyNerfedSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
        if dataset[i] != dataset[i-1]:
            datum1 = []
            for k in dataset[i-lookback:i]:
                for j in k:
                    datum1.append(j)
            try:
                dataX.append(datum1)
                dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]]))
            except:
                print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
                print("dataset[i+distance]: ", dataset[i+distance])
                # print(dataset[i+distance], "\n", dataset[i])
                print(dataset[i+distance][depth*2], dataset[i+distance][0], dataset[i][depth*2], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaCondensedSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            for j in k:
                datum1.append(j)
        try:
            dataX.append(datum1)
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]]))
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
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


# TO-DO 50%: write formatted datasets to file, read from files on training to save reformatting time
# TO-DO 100%: train in chunks to avoid overloading 8 GB GPU RAM (safe @ {Din = 30; dist = 100; perc = 0.2; c = 2} on 20 GB file)
# TO_DO 100%: mkdir models && mkdir models/models && mkdir models/training && FORMAT THE FUCKING FILEPATHS :(
# TO-DO: train inline on file stream
# TO-DO: paralellize or c-ify preprocessing
# TO-DO: tensorboard (callback)

# path = input("Enter data path:")

timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
currency_pairs, currencies = ["XBTUSD", "ETHUSD", "XRPU18", "LTCU18", "BCHU18"], ["BTCUSD", "ADABTC", "ETHUSD", "LTCBTC", "XRPBTC"]
Dfiles = ["XBTUSD02", "BMEX_XBTUSD2_100kus", "BMEX_XBTUSD3_100kus", "BMEX_XBTUSD3_10kus", "HSDBdirectionalLSTM0-Din10-dist3-lookback90-perc0.99-sparcity10-datasetBMEX_XBTUSD3_100kus.txt"]
path = Dfiles[2]
errs, Ps, passes, fails = [], [], 0, 0
Din = 10; dist = 3; perc = 0.99; c = 1.5; b = 32; nb_epoch = 15; l = 60; opt = "Adam"; s = 10
# TO-DO 50%: write formatted datasets to file, read from files on training to save reformatting time
X, Y = create_forzaSequentialDirection_dataset(forza(path, Din, perc, s, volOnly=False), dist, Din, l)
# X, Y = readDataset(path)
writeDirectionalDataset(X, Y, f'../../datasets/HSDBdirectionalLSTM0-Din{Din}-dist{dist}-lookback{l}-perc{perc}-sparcity{s}-dataset{path}')
print("X0: ", X[0], " Y0 ", Y[0], "mean/min/max(Y):", np.mean(Y), min(Y), max(Y))
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
# TO-DO 50%: write formatted datasets to file, read from files on training to save reformatting time
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=10, min_lr=0.0000000001)
saver = keras.callbacks.ModelCheckpoint(f'../models/hsdbDirectionalLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}_time{timeStr}.h5',
                                        monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
logF = f'../logs/hsdbDirectionalLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}_time{timeStr}.txt'
opt = keras.optimizers.Adam(lr=0.000666, epsilon=0.00000001, decay=0.0001, amsgrad=False)

K.tensorflow_backend._get_available_gpus()
model = Sequential()
model.add(Dense(Din*4*l, input_shape=(1, Din*4*l)))
model.add(LeakyReLU(alpha=0.666))
# model.add(Dropout(0.2))
# model.add(Dense(Din*4*l, input_shape=(1, Din*4*l)))
# model.add(Dense(Din*4*l, activation='relu'))
model.add(LSTM(Din*4*l, activation='selu', return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(Din*4*l, activation='tanh', return_sequences=False))
model.add(Dense(Din))
model.add(LeakyReLU(alpha=0.666))
# model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
model.fit_generator(hsdbSequence(trainX, trainY, b), steps_per_epoch=(len(trainX) / b),
                                          epochs=nb_epoch,
                                          verbose=2,
                                          validation_data=hsdbSequence(testX, testY, b),
                                          validation_steps=(len(testX) / b),
                                          use_multiprocessing=False,
                                          workers=8,
                                          max_queue_size=2,
                                          callbacks=[reduce_lr])
# model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=5, verbose=2, callbacks=[reduce_lr, saver])
# TO-DO 50%: write formatted datasets to file, read from files on training to save reformatting time
# FOR UNIDIMENTIONAL PREDICTIONS VVV

for i in range(len(testX)):
    sTXi = np.reshape(testX[i], (testX[i].shape[0], 1, testX[i].shape[1]))
    pY, rY = model.predict(sTXi)[0][0], testY[i]
    if (pY > 0 and rY > 0) or (pY < 0 and rY < 0):
        passes += 1
    else:
        fails += 1
    # TO-DO: INVERSE ACCURACY
    Ps.append(pY)
    errs.append(abs(pY - rY)/rY+0.00000001 * 100)
    # print("sTXi:", sTXi)
    # print("pY:", pY, "rY:", rY, "err %:", errs[-1])

print("\n\nAggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY), 
    "Mean Perc. Error:", np.mean(errs))
    
print("\nPs Describe:", sp.describe(Ps), "Ps Median:", list(sorted(Ps))[int(np.floor(len(Ps)/2))], "\nY Describe:", sp.describe(Y), "Ys Median:", list(sorted(Y))[int(np.floor(len(Y)/2))],"\ntrainY Describe:", sp.describe(trainY), "\ntestY Describe:", sp.describe(testY))

print("\n\n SAVE MODEL?")
resp = input()
if resp in ["y", "yes", "Y", "YES", "save", "SAVE"]:
    model.save(f'../models/hsdbDirectionalLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}_{timeStr}.h5')
