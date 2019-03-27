import os
import sklearn as sk
import requests, pandas as pan
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.metrics import mean_squared_error
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU, PReLU
from keras.layers import LSTM, Dropout
from keras import backend as K
import numpy as np
import time, datetime


def forza(currency="XBTUSD", depth=30, p=1):
    path = f'../HSDB_{currency}.txt'
    data, datum = [], []
    if os.path.isfile(path) == False:
        print(f'could not source {path} data')
    else:
        fileP = open(path, "r")
        lines = fileP.readlines()
        for i, line in enumerate(lines[0:int(np.floor(p * len(lines)))]):
            if i != 0 and i % 4 == 0:
                data.append(datum)
                datum = []
            linex = [float(l) for l in line.split(",")]
            datum.append(linex[0:depth])
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

def create_forzaDirection_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append(np.mean(dataset[i+1][60], dataset[i+1][0]) - np.mean(dataset[i][60], dataset[i][0]))
    return np.array(dataX), np.array(dataY)

def create_forzaFortuneTeller_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append([dataset[i+1][0], dataset[i+1][1], dataset[i+1][2], dataset[i+1][3], dataset[i+1][60], dataset[i+1][61], dataset[i+1][62], dataset[i+1][63]])
    return np.array(dataX), np.array(dataY)

timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
currency_pairs, currencies = ["XBTUSD", "ETHUSD", "XRPU18", "LTCU18", "BCHU18"], ["BTCUSD", "ADABTC", "ETHUSD", "LTCBTC", "XRPBTC"]
errs, passes, fails = [], 0, 0
Din = 30; perc = 0.5

X, Y = create_forzaDirection_dataset(forza(currency_pairs[0], Din, perc))
print("X0:\n", X[0])
print("shape(X):", X.shape)

trainX, trainY, testX, testY = X[:int(np.floor(len(X)/2))], Y[:int(np.floor(len(X)/2))], X[int(np.floor(len(X)/2)):], Y[int(np.floor(len(X)/2)):]
# print(testX[0], testY[0])

# scaler = MinMaxScaler(feature_range=(-1, 1))
# trainX = scaler.fit_transform(trainX)
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX1, (testX1.shape[0], 1, testX1.shape[1]))
# print("scaled training x[-1], y[-1]", trainX[-1], trainY[-1])
# print("trainX shape", trainX.shape)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, min_lr=0.000001)
opt = keras.optimizers.Adam(lr=0.0009, epsilon=42, decay=0.001, amsgrad=False)
K.tensorflow_backend._get_available_gpus()
model = Sequential()
model.add(Dense(Din*4, input_shape=(1, Din*4), activation='selu'))
model.add(Dense(Din*4, activation='relu'))
#model.add(LSTM(Din*2, activation='selu', return_sequences=False))
model.add(Dropout(0.05))
model.add(Dense(Din, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
model.fit(trainX, trainY, nb_epoch=10, batch_size=1, verbose=1, callbacks=[reduce_lr])
model.save(f'hsdbModel0_{timeStr}.h5')

# FOR UNIDIMENTIONAL PREDICTIONS VVV

for i in range(len(testX)):
    sTXi = testX[i]
    pY, rY = model.predict(sTXi), testY[i]
    if (pY > 0 and rY > 0) or (pY < 0 and rY < 0):
        passes += 1
    else:
        fails += 1
    errs.append((abs(pY - rY)/rY) * 100)
    print("sTXi:", sTXi)
    print("pY:", pY, "rY:", rY, "err %:", errs[-1])

print("\n\n\"Aggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY), "Mean % Error:", np.mean(errs))
