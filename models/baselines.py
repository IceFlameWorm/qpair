from __future__ import print_function
import os
import numpy as np
import pandas as pd
import pickle
import datetime, time, json
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization, Bidirectional, LSTM, dot, Flatten, Reshape, add
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import regularizers

from config import * # 在导入 baselines之前，需要在外部添加搜索路径 <project path>

def MODEL_WEM():
    DROPOUT = 0.5
    DENSEDIM = 50
    CL2 = 0.05

    with open(WEM_PKL, 'rb') as f:
        wmap = pickle.load(f)

    wem = wmap['embedding_matrix']

    question1 = Input(shape=(MAX_WSEQ_LEN,))
    question2 = Input(shape=(MAX_WSEQ_LEN,))

    dense = Dense(WORD_EMBEDDING_DIM, activation='relu')
    q1 = Embedding(WORDS_NUM + 1, WORD_EMBEDDING_DIM, weights=[wem],
                   input_length=MAX_WSEQ_LEN,
                   trainable=False)(question1)
    q1 = TimeDistributed(dense)(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(WORD_EMBEDDING_DIM,))(q1)

    q2 = Embedding(WORDS_NUM + 1, WORD_EMBEDDING_DIM, weights=[wem],
                   input_length=MAX_WSEQ_LEN,
                   trainable=False)(question2)
    q2 = TimeDistributed(dense)(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(WORD_EMBEDDING_DIM,))(q2)
    
    merged = concatenate([q1, q2]) # 顺序相关
#     merged = Dense(DENSEDIM, activation='relu', kernel_regularizer=regularizers.l2(CL2))(merged)
#     merged = Dropout(DROPOUT)(merged)
#     merged = BatchNormalization()(merged)
#     merged = Dense(DENSEDIM, activation='relu', kernel_regularizer=regularizers.l2(CL2))(merged)
#     merged = Dropout(DROPOUT)(merged)
#     merged = BatchNormalization()(merged)
#     merged = Dense(DENSEDIM, activation='relu', kernel_regularizer=regularizers.l2(CL2))(merged)
#     merged = Dropout(DROPOUT)(merged)
#     merged = BatchNormalization()(merged)
#     merged = Dense(DENSEDIM, activation='relu', kernel_regularizer=regularizers.l2(CL2))(merged)
#     merged = Dropout(DROPOUT)(merged)
#     merged = BatchNormalization()(merged)

    merged = Dense(DENSEDIM, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(DENSEDIM, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(DENSEDIM, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(DENSEDIM, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def MODEL_WEM_ATTENTION():
    DROPOUT = 0.6
    SENT_EMBEDDING_DIM = 64
    DENSEDIM = 200


    with open(WEM_PKL, 'rb') as f:
        wmap = pickle.load(f)

    wem = wmap['embedding_matrix']

    question1 = Input(shape=(MAX_WSEQ_LEN,))
    question2 = Input(shape=(MAX_WSEQ_LEN,))

    blstm = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")
    
    q1 = Embedding(WORDS_NUM + 1, WORD_EMBEDDING_DIM, weights=[wem],
                   input_length=MAX_WSEQ_LEN,
                   trainable=False)(question1)
#     q1 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q1)
    q1 = blstm(q1)

    q2 = Embedding(WORDS_NUM + 1, WORD_EMBEDDING_DIM, weights=[wem],
                   input_length=MAX_WSEQ_LEN,
                   trainable=False)(question2)
#     q2 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q2)
    q2 = blstm(q2)

    attention = dot([q1, q2], [1, 1])
    attention = Flatten()(attention)
    attention = Dense((MAX_WSEQ_LEN* SENT_EMBEDDING_DIM))(attention)
    attention = Reshape((MAX_WSEQ_LEN, SENT_EMBEDDING_DIM))(attention)

    merged = add([q1,attention])
    merged = Flatten()(merged)
    merged = Dense(DENSEDIM, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(DENSEDIM, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
#     merged = Dense(DENSEDIM, activation='relu')(merged)
#     merged = Dropout(DROPOUT)(merged)
#     merged = BatchNormalization()(merged)
#     merged = Dense(DENSEDIM, activation='relu')(merged)
#     merged = Dropout(DROPOUT)(merged)
#     merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # model = MODEL_WEM()
    model = MODEL_WEM_ATTENTION()
    print(model.summary())