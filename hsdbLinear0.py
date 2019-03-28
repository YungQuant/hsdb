import os
import time, datetime

import numpy as np
import sklearn as sk
import requests, pandas as pan
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.metrics import mean_squared_error


def forza(currency="XBTUSD", depth=30, p=1):
    path = f'../HSDB_{currency}.txt'
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

def create_forzaDirection_dataset(dataset, distance=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-distance):
        dataX.append(dataset[i])
        try:
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][60]]) - np.mean([dataset[i][0], dataset[i][60]]))
        except:
            print("create_forzaDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][60], dataset[i+distance][0], dataset[i][60], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaFortuneTeller_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append([dataset[i+1][0], dataset[i+1][1], dataset[i+1][2], dataset[i+1][3], dataset[i+1][60], dataset[i+1][61], dataset[i+1][62], dataset[i+1][63]])
    return np.array(dataX), np.array(dataY)

# TO-DO: write formatted datasets to file, read from files on training to save reformatting time
# TO-DO: train in chunks to avoid overloading 8 GB GPU RAM (safe @ {Din = 30; dist = 100; perc = 0.2; c = 2} on 20 GB file)

timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
currency_pairs, currencies = ["XBTUSD", "ETHUSD", "XRPU18", "LTCU18", "BCHU18"], ["BTCUSD", "ADABTC", "ETHUSD", "LTCBTC", "XRPBTC"]
errs, passes, fails = [], 0, 0
Din = 30; dist = 100; perc = 0.2; c = 2

X, Y = create_forzaDirection_dataset(forza(currency_pairs[0], Din, perc), dist)
print("X0: ", X[0], " Y0 ", Y[0], "mean/min/max(Y):", np.mean(Y), min(Y), max(Y))
print("\nshape(X):", X.shape)

trainX, trainY, testX, testY = X[:int(np.floor(len(X)/c))], Y[:int(np.floor(len(X)/c))], X[int(np.floor(len(X)/c)):], Y[int(np.floor(len(X)/c)):]
# print(testX[0], testY[0])

# scaler = MinMaxScaler(feature_range=(-10, 10))
# trainX = scaler.fit_transform(trainX)
# testX = scaler.fit_transform(testX)
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print("scaled training x[-1], y[-1]", trainX[-1], trainY[-1])
print("trainX shape", trainX.shape)

model = linear_model.ElasticNet()
model.fit(trainX, trainY)

# FOR UNIDIMENTIONAL PREDICTIONS VVV

for i in range(len(testX)):
    # sTXi = np.reshape(testX[i], (testX[i].shape[0], 1, testX[i].shape[1]))
    sTXi = testX[i]
    pY, rY = model.predict(sTXi.reshape(1, -1)), testY[i]
    if (pY > 0 and rY > 0) or (pY < 0 and rY < 0):
        passes += 1
    else:
        fails += 1
    errs.append(abs(pY - rY)/max([pY, rY]) * 100)
    print("sTXi:", sTXi)
    print("pY:", pY, "rY:", rY, "err %:", errs[-1])

print("\n\nAggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY), "Mean % Error:", np.mean(errs))
