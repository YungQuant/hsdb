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
import matplotlib.pyplot as plt


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
        liney = lines[i+1].split(",")[0]

        # print("linex:", linex)
        # print("liney:", liney)

        X.append([float(l) for l in linex[:-1]])
        Y.append(float(liney[:-1]))

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

def volumeAdaptiveGroupingFilter(X, g, depth, pF=True):
    Pb, Pa = X[:depth], X[depth*2:depth*3]
    Xb, Xa = list(reversed(X[depth:depth*2])), X[depth*3:]

    ix0 = 0; ix1 = 1; datumB = []
    for k in range(len(Xb)):
        if ix1 <= len(Xb):
            datumB.append(sum(Xb[ix0:ix1]))
            ix0 = ix1
            ix1 += g
        else:
            break

    ix0 = 0; ix1 = 1; datumA = []
    for k in range(len(Xa)):
        if ix1 <= len(Xa):
            datumA.append(sum(Xa[ix0:ix1]))
            ix0 = ix1
            ix1 += g
        else:
            break

    if pF == True:
        x = np.array(list(Pb)[-len(datumB):] + list(reversed(datumB)) + list(Pa)[:len(datumA)] + list(datumA))
    else:
        x = np.array(list(Pb) + list(reversed(datumB)) + list(Pa) + list(datumA))


    # plot2(Xb, datumB)
    # plot2(Xa, datumA)
    # print(f'\nXb: {Xb} \n datumB: {datumB}\nXa: {Xa} \n datumA: {datumA}\n')
    # print(f'\nX: {X} \n x: {x}\n')
    # print(f'volumeAdaptiveGroupingFilter compressed {len(X)} layers to {len(x)} layers; {(len(X)-len(x))/len(X)}%')

    return x

def volumeAdaptiveStackingFilter(X, depth, pF=True):
    Pb, Pa = X[:depth], X[depth*2:depth*3]
    Xb, Xa = list(reversed(X[depth:depth*2])), X[depth*3:]

    ix0 = 0; ix1 = 1; datumB = []
    for k in range(len(Xb)):
        if ix1 <= len(Xb):
            datumB.append(sum(Xb[ix0:ix1]))
            ix0 = ix1
            ix1 += k+1
        else:
            break

    ix0 = 0; ix1 = 1; datumA = []
    for k in range(len(Xa)):
        if ix1 <= len(Xa):
            datumA.append(sum(Xa[ix0:ix1]))
            ix0 = ix1
            ix1 += k+1
        else:
            break

    if pF == True:
        x = np.array(list(Pb)[-len(datumB):] + list(reversed(datumB)) + list(Pa)[:len(datumA)] + list(datumA))
    else:
        x = np.array(list(Pb) + list(reversed(datumB)) + list(Pa) + list(datumA))

    # plot2(Xb, datumB)
    # plot2(Xa, datumA)
    # print(f'\nXb: {Xb} \n datumB: {datumB}\nXa: {Xa} \n datumA: {datumA}\n')
    # print(f'\nX: {X} \n x: {x}\n')
    # print(f'volumeAdaptiveStackingFilter compressed {len(X)} layers to {len(x)} layers; {(len(X)-len(x))/len(X)}%')

    return x

def forza(currency="BMEX_XBTUSD2_100kus", depth=30, p=1, s=1, scale=10000, volOnly=False, vFilter="adaptive", g=0,
          pF=True, header=""):
    # path = f'../../HSDB_{currency}.txt'
    if header == "":
        path = f'../../HSDB-{currency}.txt';
        oL = 0;
        fL = 0
    else:
        path = f'../../{header}/HSDB-BMEX_{currency}.txt';
        oL = 0;
        fL = 0

    vMin = 500;
    vMax = 50000

    # /home/yungquant/HSDB-BMEX_XBTUSD2_100kus.txt
    data, datum = [], []
    if os.path.isfile(path) == False:
        print(f'\ncould not source {path} data\n')
    else:
        fileP = open(path, "r", errors="ignore")
        lines = fileP.readlines()
        lines = lines[:int(np.floor(p * len(lines)))]

        i, k, oL, nL = 0, 0, 0, 0
        while i < len(lines) - 3:
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
                    datum.append(float(list(reversed(lines[i + 1].split(",")[:depth]))[k]) / scale)
                if volOnly == False:
                    for k in range(depth):
                        datum.append(float(list(reversed(lines[i + 2].split(",")))[:depth][k]))
                for k in range(depth):
                    datum.append(float(list(reversed(lines[i + 3].split(",")))[:depth][k]) / scale)

                # if volOnly == False and datum[0] > vMin and datum[0] < vMax and (datum[depth-1] < datum[depth*3-1]):
                if datum[:depth] == sorted(datum[:depth]):
                    if vFilter == "none":
                        data.append(datum)
                    elif vFilter == "adaptive":
                        data.append(volumeAdaptiveStackingFilter(datum, depth, pF))
                    elif vFilter == "grouping":
                        data.append(volumeAdaptiveGroupingFilter(datum, g, depth, pF))

                    if i >= len(lines) - s * 10:
                        oL = len(datum)
                        nL = len(data[-1])

                else:
                    i -= 3
                datum = []

            i += 4
            k += 1
        print(f'{vFilter} filtering compressed {oL} layers to {nL} layers; {((oL-nL)/oL)*100}%')
        print(f'\nForza returning {len(data)} epochs from {currency}...\n data[-1]: {data[-1]}')

    return data

def forzaLinear(currency="BMEX_XBTUSD2_100kus", depth=30, p=1, s=1, header=""):
    # path = f'../../HSDB_{currency}.txt'
    if header == "":
        path = f'../../HSDB-{currency}.txt'; oL = 0; fL = 0
    else:
        path = f'../../{header}/HSDB-BMEX_{currency}.txt'; oL = 0; fL = 0
    # /home/yungquant/HSDB-BMEX_XBTUSD2_100kus.txt
    data, datum = [], []
    if os.path.isfile(path) == False:
        print(f'\ncould not source {path} data\n')
    else:
        fileP = open(path, "r", errors="ignore")
        lines = fileP.readlines()
        lines = lines[:int(np.floor(p * len(lines)))]

        i, k , oL, nL = 0, 0, 0, 0
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
                for k in range(depth):
                    datum.append(float(list(reversed(lines[i].split(",")[:depth]))[k]))
                for k in range(depth):
                    datum.append(float(list(reversed(lines[i+2].split(",")))[:depth][k]))

                if datum == sorted(datum):
                    data.append(datum)
                else:
                    i-=3
                datum = []

            i+=4
            k+=1
        print(f'\nForzaLinear returning {len(data)} epochs from {currency}...\n data[-1]: {data[-1]}')

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

def create_forzaSequentialLinearDirection_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []

    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            datum1.append(np.mean([k[0], k[depth]]))
        # try:
        dataX.append(datum1)
        # print(f'dataX.append({datum1})')
        # dataX.append([j for j in k for k in dataset[i-lookback:i]])
        dataY.append((np.mean([dataset[i+distance][0], dataset[i+distance][depth]]) - np.mean([dataset[i][0], dataset[i][depth]]))/np.mean([dataset[i][0], dataset[i][depth]]))
        # print(f' dataY.append({(np.mean([dataset[i+distance][0], dataset[i+distance][depth]]) - np.mean([dataset[i][0], dataset[i][depth]]))/np.mean([dataset[i][0], dataset[i][depth]])})')
        # except:
        #     print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
        #     print("dataset[i+distance]: ", dataset[i+distance])
        #     # print(dataset[i+distance], "\n", dataset[i])
        #     print(dataset[i+distance][depth], dataset[i+distance][0], dataset[i][depth], dataset[i][0])

    x, y = np.array(dataX), np.array(dataY)
    print(f'\ncreate_forzaSequentialLinearDirection_dataset returning x:{x.shape} y: {y.shape}...\n')
    return x, y

def create_forzaSequentialDirection_dataset(dataset, distance=1, depth=30, newD=10, lookback=100, vO=False, pF=True):
    dataX, dataY = [], []
    if pF == True:
        depth = newD
    dims = depth*2

    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            if vO == False:
                for j in k:
                    datum1.append(j)
            else:
                for j in k[depth:depth+newD]:
                    datum1.append(j)
                for jk in k[depth*2+newD:depth*2+newD*2]:
                    datum1.append(jk)
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][dims]]) - np.mean([dataset[i][0], dataset[i][dims]]))
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][dims], dataset[i+distance][0], dataset[i][dims], dataset[i][0])

    x, y = np.array(dataX), np.array(dataY)
    print(f'\ncreate_forzaSequentialDirection_dataset returning x:{x.shape} y: {y.shape}...\n')
    return x, y

def create_forzaSequentialLinearClassDirectionMagnitude_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []

    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            datum1.append(np.mean([k[0], k[depth]]))
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            yPrime = (np.mean([dataset[i+distance][0], dataset[i+distance][depth]]) - np.mean([dataset[i][0], dataset[i][depth]]))/np.mean([dataset[i][0], dataset[i][depth]])/2
            if yPrime > 0:
                yPrime += 0.5
                dataY.append([1-yPrime, yPrime])
            elif yPrime <= 0:
                yPrime = abs(yPrime)
                yPrime += 0.5
                dataY.append([yPrime, 1-yPrime])
        except:
            print("create_forzaSequentialLinearClassDirectionMagnitude_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][depth], dataset[i+distance][0], dataset[i][depth], dataset[i][0])

    x, y = np.array(dataX), np.array(dataY)
    print(f'\ncreate_forzaSequentialLinearClassDirectionMagnitude_dataset returning x:{x.shape} y: {y.shape}...\n')
    return x, y

def create_forzaFortuneTeller_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append([dataset[i+1][0], dataset[i+1][1], dataset[i+1][2], dataset[i+1][3], dataset[i+1][60], dataset[i+1][61], dataset[i+1][62], dataset[i+1][63]])
    return np.array(dataX), np.array(dataY)

def volatilityAdaptiveFilter(X, Y, attention=1000, interest=0.5):
    x, y = [], []
    for i in range(attention, len(Y)):
        for k in Y[i-attention:i]:
            # print(f'i:{i} , k:{k} \n Y[{i-attention}:{i}]: {Y[i-attention:i]}')
            if k >= interest or k <= -interest:
                x.append(X[i])
                y.append(Y[i])
                break
    print(f'\nvolatilityAdaptiveFilter compressed {len(Y)} epochs to {len(y)} epochs; {((len(Y)-len(y))/len(Y))*100}%\n')
    return np.array(x), np.array(y)

def volatilityAdaptiveClassFilter(X, Y, attention=1000, interest=0.5):
    x, y = [], []
    for i in range(attention, len(Y)):
        for k in Y[i-attention:i]:
            # print(f'i:{i} , k:{k} \n Y[{i-attention}:{i}]: {Y[i-attention:i]}')
            if k[0] >= interest or k[0] <= 1-interest:
                x.append(X[i])
                y.append(Y[i])
                break
    print(f'\nvolatilityAdaptiveClassFilter compressed {len(Y)} epochs to {len(y)} epochs; {((len(Y)-len(y))/len(Y))*100}%\n')
    return np.array(x), np.array(y)

def volatilitySTDAdaptiveFilter(X, Y, attention=1000, interest=0.5):
    x, y = [], []
    for i in range(attention, len(Y)):
        if np.std(Y[i-attention:i]) > interest:
            x.append(X[i])
            y.append(Y[i])
    print(f'\nvolatilitySTDAdaptiveFilter compressed {len(Y)} epochs to {len(y)} epochs; {((len(Y)-len(y))/len(Y))*100}%\n')
    return np.array(x), np.array(y)

def getEnv(header, suffix, currencies, Din, perc, s, scale, vFilter, g, pF, dist, l, vO, attention, interest):
    envSet = {}
    for currency in currencies:
        path = currency + suffix
        envSet[currency] = {"X": [], "Y": []}

        # x = forza(path, Din, perc, s, scale, False, vFilter, g, pF, header)
        # if pF == False:
        #     x, y = create_forzaSequentialDirection_dataset(x, dist, Din, int((len(x[0])-Din*2)/2), l, vO, pF)
        # elif pF == True:
        #     x, y = create_forzaSequentialDirection_dataset(x, dist, Din, int(len(x[0])/4), l, vO, pF)
        # X, Y = volatilityAdaptiveFilter(x, y, attention, interest)

        x = forzaLinear(path, Din, perc, s, header)
        # X, Y = create_forzaSequentialLinearClassDirectionMagnitude_dataset(x, dist, Din, l)
        X, Y = create_forzaSequentialLinearDirection_dataset(x, dist, Din, l)
        # X, Y = volatilityAdaptiveFilter(x, y, attention, interest)

        envSet[currency]["X"] = X
        envSet[currency]["Y"] = Y

    return envSet


def getTarget(envSet, target):
    X, Y = [], []
    keys = list(envSet.keys())

    for i in range(len(envSet[keys[0]]["X"]) - 2):
        datum = []
        for k in keys:
            for j in envSet[k]["X"][i]:
                datum.append(j)
        X.append(datum)
        Y.append(envSet[target]["Y"][i])

    return X, Y

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("")
    plt.show()

# TO-DO: write formatted datasets to file, read from files on training to save reformatting time
# TO-DO: train in chunks to avoid overloading 8 GB GPU RAM (safe @ {Din = 30; dist = 100; perc = 0.2; c = 2} on 20 GB file)

timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
header = "669c"; suffix = "_1mus"
currencies = ["XBTUSD", "ADAM19", "BCHM19", "EOSM19", "ETHUSD", "LTCM19", "TRXM19", "XRPM19"]
Dsets = ["HSDBENVdirectionalClassLSTM0-MidPeak-Linear-Din3-dist6-lookback60-perc0.99-sparcity10-attention-6-interest0.5001-header669c-targetXBTUSD.txt",
         "HSDBENVdirectionalLSTM0-MidPeak-Linear-Din3-dist6-lookback60-perc0.99-sparcity10-attention-6-interest0.005-header669c-targetXBTUSD.txt"]
target = currencies[0]
errs, Ps, passes, fails = [], [], 0, 0
Din = 3; dist = 6; perc = 0.99; c = 1.5; b = 32; nb_epoch = 50; l = 60; s = 10; scale = 10000000; vO = True
attention = 6; interest = 0.005; vFilter = "adaptive"; g = (10 if vFilter == "grouping" else 1); pF = True

# x = getEnv(header, suffix, currencies, Din, perc, s, scale, vFilter, g, pF, dist, l, vO, attention, interest)
# x, y = getTarget(x, target)
# X, Y = volatilityAdaptiveFilter(x, y, attention, interest)

X, Y = readDataset(Dsets[1])
plot(Y)
# writeDirectionalDataset(X, Y, f'../../datasets/HSDBENVdirectionalLSTM0-MidPeak-Linear-Din{Din}-dist{dist}-lookback{l}-perc{perc}-sparcity{s}-attention-{attention}-interest{interest}-header{header}-target{target}')


print(f'Set Din = {Din}; dist = {dist}; perc = {perc}; cut = {c}; lb = {l}; s = {s}')
print("Creating, formatting, & processing dataset...")

# X, Y = create_forzaSequentialDirectionClass_dataset(forza(dFiles[0], Din, perc, volOnly=False), dist, Din, lb)
print("X0: ", X[0], " Y0 ", Y[0])  # , "mean/min/max(Y):", np.mean(Y), min(Y), max(Y))
print("\nshape(X):", X.shape, "shape(Y):", Y.shape)

trainX, trainY, testX, testY = X[:int(np.floor(len(X)/c))], Y[:int(np.floor(len(X)/c))], X[int(np.floor(len(X)/c)):], Y[int(np.floor(len(X)/c)):]
# print(testX[0], testY[0])

# scaler = MinMaxScaler(feature_range=(-10, 10))
# trainX = scaler.fit_transform(trainX)
# testX = scaler.fit_transform(testX)
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print("scaled training x[-1], y[-1]", trainX[-1], trainY[-1])
print("trainX shape", trainX.shape, "trainY shape", trainY.shape)

# model = linear_model.ElasticNet(alpha=1, l1_ratio=0.5, normalize=False, precompute=False, tol=0.00001)
model = svm.SVR()
model.fit(trainX, trainY)

aPs = []; aTestYs = []; aTrainYs = []
for i in range(len(testX)):
    # sTXi = np.reshape(testX[i], (testX[i].shape[0], 1, testX[i].shape[1]))
    sTXi = np.array(testX[i]).reshape(1, -1)
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
    errs.append((abs(pY[0] - rY[0])+abs(pY[1] - rY[1]))/2*100)
    # print("sTXi:", sTXi)
    # print("pY:", pY, "rY:", rY, "err %:", errs[-1])

for y in trainY:
    maxTY = max(y)
    maxTYIx = list(y).index(maxTY)
    aTrainYs.append(maxTYIx)

print("\nPs Describe:", sp.describe(aPs), "Ps Median:", list(sorted(aPs))[int(np.floor(len(aPs)/2))], "\ntrainY Describe:", sp.describe(aTrainYs), "\ntestY Describe:", sp.describe(aTestYs))


print("\n\nAggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY),
      "Mean Perc. Error:", np.mean(errs))


