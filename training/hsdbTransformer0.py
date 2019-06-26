import argparse

import os
import datetime, math
from itertools import islice
from typing import Iterable, List, Optional

from keras import optimizers, losses
from keras.models import load_model
# noinspection PyPep8Naming
from keras import backend as K
from keras import callbacks
from keras_transformer import *
import numpy as np
from keras.utils import Sequence
from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, TransformerBlock


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

                data.append(datum)
                datum = []

            i+=4
            k+=1

    return data


def create_forzaSequentialDirection_dataset(dataset, distance=1, depth=30, lookback=100, vO=False):
    dataX, dataY = [], []
    dims = depth * 2

    for i in range(lookback, len(dataset) - distance):
        datum1 = []
        for k in dataset[i - lookback:i]:
            if vO == False:
                for j in k:
                    datum1.append(j)
            else:
                for j in k[depth:depth * 2]:
                    datum1.append(j)
                for jk in k[depth * 3:depth * 4]:
                    datum1.append(jk)
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            dataY.append(np.mean([dataset[i + distance][0], dataset[i + distance][dims]]) - np.mean(
                [dataset[i][0], dataset[i][dims]]))
        except:
            print("create_forzaSequentialDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i + distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i + distance][dims], dataset[i + distance][0], dataset[i][dims], dataset[i][0])

    return np.array(dataX), np.array(dataY)


def create_forzaSequentialClassDirection_dataset(dataset, distance=1, depth=30, lookback=100, vO=False):
    dataX, dataY = [], []
    for i in range(lookback, len(dataset) - distance):
        datum1 = []
        for k in dataset[i - lookback:i]:
            if vO == False:
                for j in k:
                    datum1.append(j)
            else:
                for j in k[depth:depth * 2]:
                    datum1.append(j)
                for jk in k[depth * 3:depth * 4]:
                    datum1.append(jk)
        try:
            dataX.append(datum1)
            # dataX.append([j for j in k for k in dataset[i-lookback:i]])
            c = (np.mean([dataset[i + distance][0], dataset[i + distance][depth * 2]]) - np.mean(
                [dataset[i][0], dataset[i][depth * 2]]))

            if c > 0:
                dataY.append([0, 0, 1])
            elif c == 0:
                dataY.append([0, 1, 0])
            elif c < 0:
                dataY.append([1, 0, 0])

        except:
            print("create_forzaSequentialClassDirection_dataset FAILED dataset[i]:", dataset[i])
            print("dataset[i+distance]: ", dataset[i + distance])
            # print(dataset[i+distance], "\n", dataset[i])
            print(dataset[i + distance][depth * 2], dataset[i + distance][0], dataset[i][depth * 2], dataset[i][0])

    return np.array(dataX), np.array(dataY)

class CosineLRSchedule:
    """
    Cosine annealing with warm restarts, described in paper
    "SGDR: stochastic gradient descent with warm restarts"
    https://arxiv.org/abs/1608.03983

    Changes the learning rate, oscillating it between `lr_high` and `lr_low`.
    It takes `period` epochs for the learning rate to drop to its very minimum,
    after which it quickly returns back to `lr_high` (resets) and everything
    starts over again.

    With every reset:
        * the period grows, multiplied by factor `period_mult`
        * the maximum learning rate drops proportionally to `high_lr_mult`

    This class is supposed to be used with
    `keras.callbacks.LearningRateScheduler`.
    """
    def __init__(self, lr_high: float, lr_low: float, initial_period: int = 50,
                 period_mult: float = 2, high_lr_mult: float = 0.97):
        self._lr_high = lr_high
        self._lr_low = lr_low
        self._initial_period = initial_period
        self._period_mult = period_mult
        self._high_lr_mult = high_lr_mult

    def __call__(self, epoch, lr):
        return self.get_lr_for_epoch(epoch)

    def get_lr_for_epoch(self, epoch):
        assert epoch >= 0
        t_cur = 0
        lr_max = self._lr_high
        period = self._initial_period
        result = lr_max
        for i in range(epoch + 1):
            if i == epoch:  # last iteration
                result = (self._lr_low +
                          0.5 * (lr_max - self._lr_low) *
                          (1 + math.cos(math.pi * t_cur / period)))
            else:
                if t_cur == period:
                    period *= self._period_mult
                    lr_max *= self._high_lr_mult
                    t_cur = 0
                else:
                    t_cur += 1
        return result

def universal_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1):
    """
    A model which is similar to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it relies L2 regularization of the word embedding matrix
    (instead of the dropout), and uses Universal Transformer architecture.
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')
    transformer_act_layer = TransformerACT(name='adaptive_computation_time')
    transformer_block = TransformerBlock(
        name='transformer', num_heads=num_heads,
        residual_dropout=transformer_dropout,
        attention_dropout=transformer_dropout,
        use_masking=True, vanilla_wiring=False)
    output_softmax_layer = Softmax(name='word_predictions')

    next_step_input, embedding_matrix = embedding_layer(word_ids)
    act_output = next_step_input

    for i in range(transformer_depth):
        next_step_input = coordinate_embedding_layer(next_step_input, step=i)
        next_step_input = transformer_block(next_step_input)
        next_step_input, act_output = transformer_act_layer(next_step_input)

    transformer_act_layer.finalize()
    next_step_input = act_output
    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    print(f'WORD_PREDICTIONS: {word_predictions} \n {word_predictions}')
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model


def vanilla_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1):
    """
    A model which is almost identical to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it uses L2 regularization of the word embedding matrix,
    instead of the dropout.
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        1,
        name='coordinate_embedding')
    output_softmax_layer = Softmax(name='word_predictions')

    next_step_input, embedding_matrix = embedding_layer(word_ids)

    next_step_input = coordinate_embedding_layer(next_step_input, step=0)
    for i in range(transformer_depth):
        next_step_input = (
            TransformerBlock(
                name='transformer' + str(i), num_heads=num_heads,
                residual_dropout=transformer_dropout,
                attention_dropout=transformer_dropout,
                use_masking=True,
                vanilla_wiring=True)
            (next_step_input))

    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model


def transformer_bert_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int,
        use_universal_transformer: bool,
        transformer_depth: int,
        num_heads: int,
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-4):
    """
    Builds a BERT-based model (Bidirectional Encoder Representations
    from Transformers) following paper "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805)

    Depending on the value passed with `use_universal_transformer` argument,
    this function applies either an Adaptive Universal Transformer (2018)
    or a vanilla Transformer (2017) to do the job (the original paper uses
    vanilla Transformer).
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    segment_ids = Input(
        shape=(max_seq_length,), dtype='int32', name='segment_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    segment_embedding_layer = Embedding(
        2,  # "Segment A" and "Segment B" embeddings
        word_embedding_size, name='segment_embeddings')
    add_segment_layer = Add(name='add_segment')
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    output_softmax_layer = Softmax(name='word_predictions')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth if use_universal_transformer else 1,
        name='coordinate_embedding')

    next_step_input, embedding_matrix = embedding_layer(word_ids)
    segment_embeddings = segment_embedding_layer(segment_ids)

    if use_universal_transformer:
        # Building a Universal Transformer (2018)
        act_layer = TransformerACT(
            name='adaptive_computation_time')
        transformer_block = TransformerBlock(
            name='transformer', num_heads=num_heads,
            residual_dropout=transformer_dropout,
            attention_dropout=transformer_dropout,
            # Allow bi-directional attention
            use_masking=False)

        act_output = next_step_input
        for i in range(transformer_depth):
            next_step_input = coordinate_embedding_layer(
                next_step_input, step=i)
            next_step_input = add_segment_layer(
                [next_step_input, segment_embeddings])
            next_step_input = transformer_block(next_step_input)
            next_step_input, act_output = act_layer(next_step_input)

        act_layer.finalize()
        next_step_input = act_output
    else:
        # Building a Vanilla Transformer (described in
        # "Attention is all you need", 2017)
        next_step_input = coordinate_embedding_layer(next_step_input, step=0)
        next_step_input = add_segment_layer(
            [next_step_input, segment_embeddings])
        for i in range(transformer_depth):
            next_step_input = (
                TransformerBlock(
                    name='transformer' + str(i), num_heads=num_heads,
                    residual_dropout=transformer_dropout,
                    attention_dropout=transformer_dropout,
                    use_masking=False,  # Allow bi-directional attention
                    vanilla_wiring=True)
                (next_step_input))

    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    cls_node_slice = (
        # selecting the first output position in each sequence
        # (responsible for classification)
        Lambda(lambda x: x[:, 0], name='cls_node_slicer')
        (next_step_input))
    class_prediction = (
        Dense(1, name='class_prediction', activation='sigmoid')
        (cls_node_slice))
    model = Model(
        inputs=[word_ids, segment_ids],
        outputs=[word_predictions, class_prediction])
    return model



# def main(model_save_path: str,
#          model_name: str,
#          tensorboard_log_path: Optional[str],
#          num_epochs: int,
#          learning_rate: float,
#          batch_size: int,
#          max_seq_length: int,
#          word_embedding_size: int,
#          load_weights_only: bool,
#          show_model_summary: bool):
#     contain_tf_gpu_mem_usage()
#     encoder = wikitext.build_wikitext_bpe_encoder()

# def x_y_for_dataset(dataset_name):
#     fat_sample = training_data_to_dense_samples(
#         dataset_name, encoder, max_seq_length)
#     _x = fat_sample[:, :max_seq_length]
#     _y = np.expand_dims(fat_sample[:, 1:], axis=-1)
#     return _x, _y



def compile_new_model(model_name, learning_rate, max_seq_length, vocabulary_size, word_embedding_size, metric, transformer_depth, num_heads):
    if model_name == 'universal':
        optimizer = optimizers.Adam(
            lr=learning_rate, beta_1=0.6, beta_2=0.999)
        _model = universal_transformer_gpt_model(
            max_seq_length=max_seq_length,
            vocabulary_size=vocabulary_size,
            word_embedding_size=word_embedding_size,
            transformer_depth=transformer_depth,
            num_heads=num_heads)
        _model.compile(
            optimizer,
            loss=losses.sparse_categorical_crossentropy,
            metrics=metric)
    elif model_name == 'vanilla':
        optimizer = optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
        _model = vanilla_transformer_gpt_model(
            max_seq_length=max_seq_length,
            vocabulary_size=vocabulary_size,
            word_embedding_size=word_embedding_size,
            transformer_depth=transformer_depth,
            num_heads=num_heads)
        _model.compile(
            optimizer,
            loss=losses.sparse_categorical_crossentropy,
            metrics=metric)
    else:
        raise RuntimeError(f'Unknown model {model_name}')
    return _model

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


timeStr = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
currency_pairs, currencies = ["XBTUSD", "ETHUSD", "XRPU18", "LTCU18", "BCHU18"], ["BTCUSD", "ADABTC", "ETHUSD", "LTCBTC", "XRPBTC"]
Dfiles = ["XBTUSD02", "BMEX_XBTUSD2_100kus", "BMEX_XBTUSD3_100kus", "BMEX_XBTUSD3_10kus",
"HSDBdirectionalLSTM0-Din20-dist3-lookback30-perc0.99-sparcity10-datasetBMEX_XBTUSD2_100kus-time2019-04-20T02:40:32.644265UTC__2019-04-20T02:45:14.197736UTC.txt",
"HSDBdirectionalClassLSTM0-MidPeak-Din20-dist10-lookback20-perc0.99-sparcity10-scale100000-vOTrue-datasetBMEX_XBTUSD3_100kus.txt",
 "BMEX_XBTUSD5_10kus"]
path = Dfiles[1]
# errs, Ps, aPs, aTestYs, aTrainYs, passes, fails = [], [], [], [], [], 0, 0
# Din = 20; dist = 10; perc = 0.99; c = 1.5; b = 32; nb_epoch = 50; l = 20; opt = "Adadelta"; s = 10; scale = 100000; vO = True


model_save_path = "not_a_real_file.h5"
load_weights_only = False; show_model_summary = True
learning_rate = 0.0666; num_epochs = 50; b = 32; c = 1.5; Din = 5; dist = 10; perc = 0.1; s = 10; scale = 100000; l = 10; vO = True; transformer_depth = 5; num_heads = 5
model_name = "universal"; learning_rate = learning_rate; max_seq_length = Din*2*l; vocabulary_size = 10; word_embedding_size = 5; metric = ["accuracy"]
# def compile_new_model(model_name, learning_rate, max_seq_length, encoder, word_embedding_size, metric):
# word_ids (InputLayer) = (none, max_seq_length)
# bpe_embeddings (ReusableEmbedding)  = (none, max_seq_length/word_ids, word_embedding_size), (vocabulary_size, word_embedding_size)

if os.path.exists(model_save_path):
    if load_weights_only:
        print('Loading the whole model from', model_save_path)
        model = load_model(model_save_path)
else:
    model = compile_new_model(model_name, learning_rate, max_seq_length, vocabulary_size, word_embedding_size, metric, transformer_depth, num_heads)

if show_model_summary:
    model.summary(120)

lr_scheduler = callbacks.LearningRateScheduler(
    CosineLRSchedule(lr_high=learning_rate,
                     lr_low=learning_rate / 32,
                     initial_period=num_epochs),
    verbose=1)
checkpoint = callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_loss', save_best_only=True, verbose=True)

model_callbacks = [
    # checkpoint,
    # lr_scheduler,
]

X, Y = create_forzaSequentialDirection_dataset(forza(path, Din, perc, s, scale, volOnly=False), dist, Din, l, vO=vO)
# writeDirectionalDataset(X, Y,
#                         f'../../datasets/HSDBdirectionalLSTM0-Din{Din}-dist{dist}-perc{perc}-sparcity{s}-dataset{path}-lookback{l}-_thispartgetscut')
print("X0: ", X[0], " Y0 ", Y[0], "mean/min/max(Y):", np.mean(Y), min(Y), max(Y))
# print("\nshape(X):", X.shape)

trainX, trainY, testX, testY = X[:int(np.floor(len(X) / c))], Y[:int(np.floor(len(X) / c))], X[int(
    np.floor(len(X) / c)):], Y[int(np.floor(len(X) / c)):]

# if tensorboard_log_path:
#     model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path))

model.fit(
    trainX, trainY,
    batch_size=b, epochs=num_epochs,
    callbacks=model_callbacks)

# model.fit_generator(hsdbSequence(trainX, trainY, b), steps_per_epoch=(len(trainX) / b),
#                                       epochs=num_epochs,
#                                       verbose=2,
#                                       validation_data=hsdbSequence(testX, testY, b),
#                                       validation_steps=(len(testX) / b),
#                                       use_multiprocessing=False,
#                                       workers=8,
#                                       max_queue_size=2,
#                                       callbacks=model_callbacks)

# Evaluation using test set
print('-' * 80)
test_x, test_y = testX, testY
test_metrics = model.evaluate(test_x, test_y, batch_size=b)
for metric_name, metric_value in zip(model.metrics_names, test_metrics):
    print(f'Test {metric_name}:', metric_value)