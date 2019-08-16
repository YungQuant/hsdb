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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU, PReLU
from keras.layers import LSTM, Dropout
from keras import backend as K
import numpy as np
import time, datetime

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("")
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    y2 = np.arange(len(b))
    plt.plot(y, a, 'g', y2, b, 'r')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("")
    plt.show()

def plot3(a, b, c):
    y = np.arange(len(a))
    y1 = np.arange(len(b))
    y2 = np.arange(len(c))
    plt.plot(y, a, 'g', y1, b, 'r', y2, c, 'b')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title('')
    plt.show()

def writeDirectionalDataset(X, Y, path="../..HSDB_unnamedDataset.txt"):
    path = path + ".txt"

    if os.path.isfile(path) == True:
        print(f'\nFound {path}, not reproducing\n')
        return
    else:
        print(f'\nWriteing {path}...\n')

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

def readDataset(path="../..HSDB_unnamedDataset.txt"):
    path = "../../datasets/" + path
    print("Loading: ", path)
    if os.path.isfile(path) == False:
        print(f'{path}, not found')
    fileP = open(path, "r")
    lines = fileP.readlines()

    X, Y = [], []

    i = 0
    while i < len(lines)-1:
        linex = lines[i].split(",")
        liney = lines[i+1].split(",")

        X.append([float(l) for l in linex[:]])
        Y.append([float(l) for l in liney[:]])

        i += 2

    return np.array(X), np.array(Y)

def readDirectionalClassDataset(path="../..HSDB_unnamedDataset.txt", old=False):
    path = "../../datasets/" + path
    print("Loading: ", path)
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

def forza(currency="BMEX_XBTUSD2_100kus", depth=30, p=1, s=1, scale=10000, volOnly=False, vFilter="adaptive", g=0, pF=True, header=""):
    # path = f'../../HSDB_{currency}.txt'
    if header == "":
        path = f'../../HSDB-{currency}.txt'; oL = 0; fL = 0
    else:
        path = f'../../{header}/HSDB-BMEX_{currency}.txt'; oL = 0; fL = 0

    vMin = 500; vMax = 50000

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

                # if volOnly == False and datum[0] > vMin and datum[0] < vMax and (datum[depth-1] < datum[depth*3-1]):
                if datum[:depth] == sorted(datum[:depth]):
                    if vFilter == "none":
                        data.append(datum)
                    elif vFilter == "adaptive":
                        data.append(volumeAdaptiveStackingFilter(datum, depth, pF))
                    elif vFilter == "grouping":
                        data.append(volumeAdaptiveGroupingFilter(datum, g, depth, pF))
                    
                    if i >= len(lines) - s*10:
                        oL = len(datum)
                        nL = len(data[-1])
                        
                else:
                    i-=3
                datum = []

            i+=4
            k+=1
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

def create_forzaSequentialLinearDirection_dataset(dataset, distance=1, depth=30, lookback=100):
    dataX, dataY = [], []

    for i in range(lookback, len(dataset)-distance):
        datum1 = []
        for k in dataset[i-lookback:i]:
            datum1.append(np.mean([k[0], k[depth]]))
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            dataY.append((np.mean([dataset[i+distance][0], dataset[i+distance][depth]]) - np.mean([dataset[i][0], dataset[i][depth]]))/np.mean([dataset[i][0], dataset[i][depth]]))
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][depth], dataset[i+distance][0], dataset[i][depth], dataset[i][0])

    x, y = np.array(dataX), np.array(dataY)
    print(f'\ncreate_forzaSequentialLinearDirection_dataset returning x:{x.shape} y: {y.shape}...\n')
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

def create_forzaSequentialClassDirectionMagnitude_dataset(dataset, distance=1, depth=30, newD=10, lookback=100, vO=False, pF=True):
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
            yPrime = (np.mean([dataset[i+distance][0], dataset[i+distance][depth]]) - np.mean([dataset[i][0], dataset[i][depth]]))/np.mean([dataset[i][0], dataset[i][depth]])/2
            if yPrime > 0:
                yPrime += 0.5
                dataY.append([1-yPrime, yPrime])
            elif yPrime <= 0:
                yPrime = abs(yPrime)
                yPrime += 0.5
                dataY.append([yPrime, 1-yPrime])
        except:
            print("create_forzaSequentialClassDirectionMagnitude_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][dims], dataset[i+distance][0], dataset[i][dims], dataset[i][0])

    x, y = np.array(dataX), np.array(dataY)
    print(f'\ncreate_forzaSequentialClassDirectionMagnitude_dataset returning x:{x.shape} y: {y.shape}...\n')
    return x, y

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



def create_forzaFortuneTeller_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append([dataset[i+1][0], dataset[i+1][1], dataset[i+1][2], dataset[i+1][3], dataset[i+1][60], dataset[i+1][61], dataset[i+1][62], dataset[i+1][63]])
    return np.array(dataX), np.array(dataY)

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

def getEnv(header, suffix, currencies, Din, perc, s, scale, vFilter, g, pF, dist, l, vO):
    envSet = {}
    for currency in currencies:
        path = currency + suffix
        envSet[currency] = {"X": [], "Y": []}

        V VOLUME V
        x = forza(path, Din, perc, s, scale, False, vFilter, g, pF, header)
        if pF == False:
            x, y = create_forzaSequentialDirection_dataset(x, dist, Din, int((len(x[0])-Din*2)/2), l, vO, pF)
        elif pF == True:
            X, Y = create_forzaSequentialDirection_dataset(x, dist, Din, int(len(x[0])/4), l, vO, pF)

        # V PRICE V
        # x = forzaLinear(path, Din, perc, s, header)
        # X, Y = create_forzaSequentialLinearClassDirectionMagnitude_dataset(x, dist, Din, l)

        envSet[currency]["X"] = X
        envSet[currency]["Y"] = Y

    return envSet


def getTarget(envSet, target):
    X, Y = [], []
    keys = list(envSet.keys())
    
    for i in range(len(envSet[keys[0]]["X"])-2):
        datum = []
        for k in keys:
            for j in envSet[k]["X"][i]:
                datum.append(j)
        X.append(datum)
        Y.append(envSet[target]["Y"][i])
    
    return X, Y



# TO-DO: train inline on file stream
# TO-DO: paralellize or c-ify preprocessing
# TO-DO: tensorboard (callback)

# path = input("Enter data path:")

timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
header = "669c"; suffix = "_1mus"
currencies = ["XBTUSD", "ADAM19", "BCHM19", "EOSM19", "ETHUSD", "LTCM19", "TRXM19", "XRPM19"]
Dsets = ["HSDBENVdirectionalClassMagnitudeLSTM0-MidPeak-Linear-Din3-dist1-lookback3-perc0.99-sparcity10-attention-60-interest0.5001-header669c-targetXBTUSD.txt"]
target = currencies[0]
errs, Ps, aPs, aTestYs, aTrainYs, passes, fails = [], [], [], [], [], 0, 0
Din = 3; dist = 1; perc = 0.99; c = 1.5; b = 32; nb_epoch = 50; l = 3; opt = "Adadelta"; kInit = "glorotUniform"; bInit = "zeros"; s = 10; scale = 10000000; vO = True
attention = 60; interest = 0.5001; vFilter = "adaptive"; g = (10 if vFilter == "grouping" else 1); pF = True

x = getEnv(header, suffix, currencies, Din, perc, s, scale, vFilter, g, pF, dist, l, vO)
x, y = getTarget(x, target)
X, Y = volatilityAdaptiveClassFilter(x, y, attention, interest)

# X, Y = readDirectionalClassDataset(Dsets[0])
plot([g[0] for g in Y])
writeDirectionalClassDataset(X, Y, f'../../datasets/HSDBENVdirectionalClassMagnitudeLSTM0-MidPeak-Linear-Din{Din}-dist{dist}-lookback{l}-perc{perc}-sparcity{s}-attention-{attention}-interest{interest}-header{header}-target{target}')
# writeDirectionalDataset(X, Y, f'../../datasets/HSDBdirectionalLSTM0-MidPeak-VolAdaptive-Din{Din}-dist{dist}-lookback{l}-perc{perc}-sparcity{s}-scale{scale}-vO{vO}-pF{pF}-attention-{attention}-interest{interest}-vFilter{vFilter}-grouping{g}-dataset{path}')
print("X0: ", X[0], " Y0 ", Y[0])
# print("\nmean/min/max(Y):", (np.mean(Y[:][0]) + np.mean(y[:][1]))/2, min(Y[:][0]), max(Y[:][1]), "\n")
#print("\nshape(X):", X.shape)

trainX, trainY, testX, testY = X[:int(np.floor(len(X)/c))], Y[:int(np.floor(len(X)/c))], X[int(np.floor(len(X)/c)):], Y[int(np.floor(len(X)/c)):]
# print(testX[0], testY[0])

# scaler = MinMaxScaler(feature_range=(-10, 10))
# trainX = scaler.fit_transform(trainX)
# testX = scaler.fit_transform(testX)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print("scaled training x[-1], y[-1]", trainX[-1], trainY[-1])
print("trainX shape", trainX.shape, "testX shape", testX.shape, "\n")
dims = trainX.shape[2]
print(f'Set DIMS: {dims}\n')

# lrs = LearningRateScheduler(cosl)
stop = EarlyStopping(patience=10)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=10, min_lr=0.0000000001)
# saver = keras.callbacks.ModelCheckpoint(f'../models/hsdbDirectionalLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}_time{timeStr}.h5',
#                                         monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
logF = f'../logs/hsdbDirectionalLSTMModel0-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}_time{timeStr}.txt'
# opt = keras.optimizers.Adam(lr=0.00000666, epsilon=0.00000001, decay=0.001, amsgrad=True)
# opt = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# opt = keras.optimizers.Adagrad(lr=0.000001, epsilon=None, decay=0.666)
opt = keras.optimizers.Adadelta(lr=0.000001, rho=0.95, epsilon=None, decay=0.1)
# opt = keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

K.tensorflow_backend._get_available_gpus()
model = Sequential()
model.add(Dense(dims, input_shape=(1, dims)))
# model.add(Dense(dims, input_shape=(1, dims), kernel_initializer=kInit, bias_initializer=bInit))  #CHANGE MODEL SAVE FILE NAME IF CUSTOM INIT
model.add(LeakyReLU(alpha=0.2))
# model.add(Dropout(0.2))
# model.add(Dense(dims*4))
# model.add(LeakyReLU(alpha=0.4))
# model.add(Dropout(0.3))
# model.add(LSTM(dims*4, return_sequences=True))
# model.add(LeakyReLU(alpha=0.2))
# model.add(Dropout(0.3))
model.add(LSTM(dims, return_sequences=False))
model.add(LeakyReLU(alpha=0.1))
# model.add(Dense(int(np.floor(dims/4))))
# model.add(LeakyReLU(alpha=0.6))
# model.add(Flatten())
model.add(Dense(2, activation='softmax'))
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
    errs.append((abs(pY[0] - rY[0])+abs(pY[1] - rY[1]))/2*100)
    # print("sTXi:", sTXi)
    # print("pY:", pY, "rY:", rY, "err %:", errs[-1])

print("\n\nAggregate Binary Accuracy:", passes, "/", len(testY), "ABA%:", passes / len(testY), 
    "Mean Perc. Error:", np.mean(errs))

for y in trainY:
    maxTY = max(y)
    maxTYIx = list(y).index(maxTY)
    aTrainYs.append(maxTYIx)

print("\nPs Describe:", sp.describe(aPs), "Ps Median:", list(sorted(aPs))[int(np.floor(len(aPs)/2))], "\ntrainY Describe:", sp.describe(aTrainYs), "\ntestY Describe:", sp.describe(aTestYs))


print(f'\nhsdbENVDirectionalClassMagnitudeLSTMModel0-target{target}-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}-mpe{np.mean(errs)}-aba{passes / len(testY)}-{timeStr}.h5')
print("\n\n SAVE MODEL?")
resp = input()
if resp in ["y", "yes", "Y", "YES", "save", "SAVE"]:
    model.save(f'../models/hsdbENVDirectionalClassMagnitudeLSTMModel0-target{target}-Din{Din}-dist{dist}-lookback{l}-batch{b}-opt{opt}-epoch{nb_epoch}-mpe{np.mean(errs)}-aba{passes / len(testY)}-{timeStr}.h5')
