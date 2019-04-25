import os
import numpy as np
import matplotlib.pyplot as plt

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

                data.append(datum)
                datum = []

            i+=4
            k+=1

    return data

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

def create_forzaSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100, vO=False):
    dataX, dataY = [], []
    dims = depth*2

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
            dataY.append(np.mean([dataset[i+distance][0], dataset[i+distance][dims]]) - np.mean([dataset[i][0], dataset[i][dims]]))
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i+distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i+distance][dims], dataset[i+distance][0], dataset[i][dims], dataset[i][0])

    return np.array(dataX), np.array(dataY)

def plot(a):
    y = np.arange(len(a))
    plt.plot(y, a, 'g')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("")
    plt.show()

def plot2(a, b):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("")
    plt.show()

def plot3(a, b, c):
    y = np.arange(len(a))
    plt.plot(y, a, 'g', y, b, 'r', y, c, 'b')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title('')
    plt.show()


Dfiles = ["XBTUSD02", "BMEX_XBTUSD2_100kus", "BMEX_XBTUSD3_100kus", "BMEX_XBTUSD3_10kus", "BMEX_XBTUSD5_10kus"]
X = []
path = Dfiles[-1]
x, Y = create_forzaSequentialDirection_dataset(forza(currency=path, depth=10, p=1, s=10, scale=100000000, volOnly=False), distance=100, depth=10, lookback=10, vO=True)

for k in x:
    X.append(sum(k))

plot2(X, Y)


# path = "../../datasets/HSDBdirectionalLSTM0-Din10-dist3-lookback90-perc0.99-sparcity10-datasetBMEX_XBTUSD3_100kus.txt"
# # /home/yungquant/datasets/HSDBdirectionalLSTM0-Din10-dist3-lookback90-perc0.99-sparcity10-datasetBMEX_XBTUSD3_100kus.txt
# if os.path.isfile(path) == False:
#     print(f'could not source {path} data')
# else:
#     fileP = open(path, "r")
#     lines = fileP.readlines(100000)
#     for i, line in enumerate(lines):
#         print(f'line {i}: {line[:1000]}')
#     # [print(line[:100], "\n") for line in lines]
        
    
