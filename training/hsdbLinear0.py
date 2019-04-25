import os
import time, datetime

import numpy as np
import sklearn as sk
import requests, pandas as pan
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.linear_model           import LogisticRegression
from sklearn.tree                   import DecisionTreeClassifier
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.naive_bayes            import GaussianNB
import scipy.stats as sp
from sklearn.metrics import mean_squared_error


def writeDirectionalDataset(X, Y, path="../..HSDB_unnamedDataset.txt"):
    path = path.split("_")[0] + "_" + str(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")) + ".txt"

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

        i += 2

    return X, Y

def forza(currency="XBTUSD", depth=30, p=1.0, volOnly=False):
    # path = f'../../HSDB-BMEX_{currency}3_100kus.txt'
    # HSDB - BMEX_XBTUSD2_10kus.txt
    path = currency
    data, datum = [], []
    if os.path.isfile(path) == False:
        print(f'could not source {path} data')
    else:
        fileP = open(path, "r")
        lines = fileP.readlines()
        lines = lines[0:int(np.floor(p * len(lines)))]

        i = 0
        while i < len(lines)-4:
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

            i += 4

    return data

def classify(x):
    return int(x*2)

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

def create_forzaSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
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

def create_forzaSequentialDirectionClass_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            for j in k:
                datum1.append(j)
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            cls = np.mean([dataset[i+distance][0], dataset[i+distance][depth*2]]) - np.mean([dataset[i][0], dataset[i][depth*2]])
            dataY.append(classify(cls))
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][depth*2], dataset[i+distance][0], dataset[i][depth*2], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaSparseSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100, sparcity=10, volOnly=False):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset)-distance-1):
        if i % sparcity == 0:

            datum1 = []
            for k in dataset[i-lookback:i]:
                for j in k[depth:depth*2] + k[depth*3:]:
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

def create_forzaSequentialLinearDirection_dataset(dataset, din, distance=1, lookback=100, volOnly=False):
    dataX, dataY, datum = [], [], []
    for i in range(lookback, len(dataset)-distance):
        if volOnly == False:
            dinX = din*2
        else:
            dinX = din

        for k in dataset[i-lookback:i]:
            datum.append(np.mean([k[0], k[dinX]]))
        try:
            dataX.append(datum)
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][dinX]]) - np.mean([dataset[i][0], dataset[i][dinX]]))
            datum = []
        except:
            print("create_forzaSequentialLinearDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][dinX], dataset[i+distance][0], dataset[i][dinX], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def create_forzaSparseSequentialLinearDirection_dataset(dataset, din, distance=1, lookback=100, sparcity=10, volOnly=False):
    dataX, dataY, datum = [], [], []
    for i in range(lookback, len(dataset)-distance):
        if i % sparcity == 0:
            if volOnly == False:
                dinX = din * 2
            else:
                dinX = din

            for k in dataset[i-lookback:i]:
                datum.append(np.mean([k[0], k[dinX]]))
            try:
                dataX.append(datum)
                dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][dinX]]) - np.mean([dataset[i][0], dataset[i][dinX]]))
                datum = []
            except:
                print("create_forzaSequentialLinearDirection_dataset FAILED dataset[i]:", dataset[i])
                print("dataset[i+distance]: ", dataset[i+distance])
                print(dataset[i+distance], "\n", dataset[i])
                print(dataset[i+distance][dinX], dataset[i+distance][0], dataset[i][dinX], dataset[i][0])

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
dFiles = ["../../HSDB-BMEX_XBTUSD3_100kus.txt"]
Ps, errs, passes, fails = [], [], 0, 0
Din = 30; dist = 10; perc = 0.99; c = 1.5; lb = 100; s = 100

print(f'Set Din = {Din}; dist = {dist}; perc = {perc}; cut = {c}; lb = {lb}; s = {s}')
print("Creating, formatting, & processing dataset...")

X, Y = create_forzaSequentialDirectionClass_dataset(forza(dFiles[0], Din, perc, volOnly=False), dist, Din, lb)
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

# model = linear_model.ElasticNet(alpha=1, l1_ratio=0.5, normalize=False, precompute=False, tol=0.00001)
model = LogisticRegression()
model.fit(trainX, trainY)

# FOR UNIDIMENTIONAL PREDICTIONS VVV

for i in range(len(testX)):
    sTXi = np.array(testX[i]).reshape(1, -1)
    pY, rY = model.predict(sTXi)[0], testY[i]
    if (pY > 0 and rY > 0) or (pY < 0 and rY < 0):
        passes += 1
    else:
        fails += 1
    Ps.append(pY)
    errs.append(abs(pY - rY)/(rY+0.00000001) * 100)
    # print("sTXi:", sTXi)
    print("pY:", pY, "rY:", rY, "err %:", errs[-1])

print("\n\nAggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY),
      "Mean Perc. Error:", np.mean(errs))

print("\nPs Describe:", sp.describe(Ps), "\nY Describe:", sp.describe(Y), "\ntrainY Describe:", sp.describe(trainY),
      "\ntestY Describe:", sp.describe(testY))
