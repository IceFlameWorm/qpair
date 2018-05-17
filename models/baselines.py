from __future__ import print_function
import os
import numpy as np
import pandas as pd
import pickle
import datetime, time, json
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split

from config import * # 在导入 baselines之前，需要在外部添加搜索路径 <project path>

def MODEL_WEM():
    DROPOUT = 0.1


    with open(WEM_PKL, 'rb') as f:
        wamp = pickle.load(f)

    wem = wamp['embedding_matrix']

    question1 = Input(shape=(MAX_WSEQ_LEN,))
    question2 = Input(shape=(MAX_WSEQ_LEN,))

    q1 = Embedding(WORDS_NUM + 1, WORD_EMBEDDING_DIM, weights=[wem],
                   input_length=MAX_WSEQ_LEN,
                   trainable=False)(question1)
    q1 = TimeDistributed(Dense(WORD_EMBEDDING_DIM, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(WORD_EMBEDDING_DIM,))(q1)

    q2 = Embedding(WORDS_NUM + 1, WORD_EMBEDDING_DIM, weights=[wem],
                   input_length=MAX_WSEQ_LEN,
                   trainable=False)(question2)
    q2 = TimeDistributed(Dense(WORD_EMBEDDING_DIM, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(WORD_EMBEDDING_DIM,))(q2)

    merged = concatenate([q1, q2]) # 顺序相关
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model_wem = MODEL_WEM()
    print(model_wem.summary())