import os
import sklearn as sk
import requests, pandas as pan
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.metrics import mean_squared_error
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
    path = path.split("_")[0] + "_" + str(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")) + ".txt"
    fileP = open(path, "w")

    for i in range(len(X)):
        try:
            strX, strY = "", ""
            for k in range(len(X[i])):
                strX += str(X[i][k])
                strX += ","

            strY = str(Y[i])

            fileP.write(strX)
            fileP.write(strY)

        except Exception as e:
            print(e)

    fileP.close()

def readDataset(path="../..HSDB_unnamedDataset.txt"):
    fileP = open(path, "r")
    lines = fileP.readlines()

    X, Y = [], []

    i = 0
    while i < len(lines)-1:
        linex = lines[i].split(",")
        liney = lines[i+1].split(",")

        X.append([float(l) for l in linex])
        Y.append([float(l) for l in liney])

        i+=2

    return X, Y

def forza(currency="XBTUSD", depth=30, p=1):
    path = f'../../HSDB_{currency}.txt'
    data, datum = [], []
    if os.path.isfile(path) == False:
        print(f'could not source {path} data')
    else:
        fileP = open(path, "r")
        lines = fileP.readlines()
        lines = lines[0:int(np.floor(p * len(lines)))]

        i = 0
        while i < len(lines)-3:
            bidP = lines[i].split(",")[:depth]
            bidV = lines[i+1].split(",")[:depth]
            askP = list(reversed(lines[i+2].split(",")))[:depth]
            askV = list(reversed(lines[i+3].split(",")))[:depth]

            for k in range(depth):
                datum.append(float(bidP[k]))
            for k in range(depth):
                datum.append(float(bidV[k]))
            for k in range(depth):
                datum.append(float(askP[k]))
            for k in range(depth):
                datum.append(float(askV[k]))

            data.append(datum)
            datum = []

            i+=4

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

# TO-DO: write formatted datasets to file, read from files on training to save reformatting time
# TO-DO: train in chunks to avoid overloading 8 GB GPU RAM (safe @ {Din = 30; dist = 100; perc = 0.2; c = 2} on 20 GB file)
# TO_DO: mkdir models && mkdir models/models && mkdir models/training && FORMAT THE FUCKING FILEPATHS :(

path = input("Enter data path:")

timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
currency_pairs, currencies = ["XBTUSD", "ETHUSD", "XRPU18", "LTCU18", "BCHU18"], ["BTCUSD", "ADABTC", "ETHUSD", "LTCBTC", "XRPBTC"]
Dfiles = ["XBTUSD02"]
errs, Ps, passes, fails = [], [], 0, 0
Din = 10; dist = 666; perc = 1; c = 1.05; b = 1000; nb_epoch = 25

X, Y = create_forzaDirection_dataset(forza(path, Din, perc), dist, Din)
writeDirectionalDataset(X, Y, f'../../HSDBdirectionalFF0-Din{Din}-dist{dist}-perc{perc}-cut{c}-dataset{path}-_thispartgetscut')
print("X0: ", X[0], " Y0 ", Y[0], "mean/min/max(Y):", np.mean(Y), min(Y), max(Y))
print("\nshape(X):", X.shape)

trainX, trainY, testX, testY = X[:int(np.floor(len(X)/c))], Y[:int(np.floor(len(X)/c))], X[int(np.floor(len(X)/c)):], Y[int(np.floor(len(X)/c)):]
# print(testX[0], testY[0])

# scaler = MinMaxScaler(feature_range=(-10, 10))
# trainX = scaler.fit_transform(trainX)
# testX = scaler.fit_transform(testX)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print("scaled training x[-1], y[-1]", trainX[-1], trainY[-1])
print("trainX shape", trainX.shape)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=10, min_lr=0.0000000001)
saver = keras.callbacks.ModelCheckpoint(f'../models/hsdbDirectionalFFModel0_{timeStr}.h5', monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#opt = keras.optimizers.Adam(lr=0.0005, epsilon=0.00000001, decay=0.00001, amsgrad=False)
opt = "Adam"
K.tensorflow_backend._get_available_gpus()
model = Sequential()
model.add(Dense(Din*4, input_shape=(1, Din*4), activation='selu'))
model.add(Dense(Din*4, activation='relu'))
#model.add(LSTM(Din*2, activation='selu', return_sequences=False))
model.add(Dropout(0.33))
model.add(Dense(Din, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
model.fit_generator(hsdbSequence(trainX, trainY, b), steps_per_epoch=(len(trainX) / b),
                                          epochs=nb_epoch,
                                          verbose=2,
                                          validation_data=hsdbSequence(testX, testY, b),
                                          validation_steps=(len(testX) / b),
                                          use_multiprocessing=False,
                                          workers=4,
                                          max_queue_size=4,
                                          callbacks=[reduce_lr, saver])
# model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=5, verbose=2, callbacks=[reduce_lr, saver])

# FOR UNIDIMENTIONAL PREDICTIONS VVV

for i in range(len(testX)):
    sTXi = np.reshape(testX[i], (testX[i].shape[0], 1, testX[i].shape[1]))
    pY, rY = model.predict(sTXi)[0][0], testY[i]
    if (pY > 0 and rY > 0) or (pY < 0 and rY < 0):
        passes += 1
    else:
        fails += 1
    Ps.append(pY)
    errs.append(abs(pY - rY)/max([pY, rY]) * 100)
    print("sTXi:", sTXi)
    print("pY:", pY, "rY:", rY, "err %:", errs[-1])

print("\n\nAggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY), "Mean Perc. Error:", np.mean(errs), "Mean Pred.:", np.mean(Ps))
