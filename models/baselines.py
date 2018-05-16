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

DATA_SET_PATH = "/root/mounted/datasets/data_0515/"
KN_CSV = os.path.join(DATA_SET_PATH, "knowledge.csv")
TRAIN_CSV = os.path.join(DATA_SET_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_SET_PATH, "test.csv")
SUBMIT_CSV = os.path.join(DATA_SET_PATH, "submit.csv")
CHAR_EMBED_PKL = os.path.join(DATA_SET_PATH, "char_embed.pkl")
WORD_EMBED_PKL = os.path.join(DATA_SET_PATH, "word_embed.pkl")
QUESTION_PKL = os.path.join(DATA_SET_PATH, "question.pkl")
INTERMEDIATE_DATA_PATH = os.path.join(DATA_SET_PATH, 'intermediate')
KN_TRAIN_CSV = os.path.join(INTERMEDIATE_DATA_PATH, 'kn_train.csv')
KN_TRAIN_WV_CV_CSV = os.path.join(INTERMEDIATE_DATA_PATH, 'kn_train_wv_cv.csv')
TEST_WV_CV_CSV = os.path.join(INTERMEDIATE_DATA_PATH, 'test_wv_cv.csv')
WEM_PKL = os.path.join(INTERMEDIATE_DATA_PATH, 'WEM.pkl')
CEM_PKL = os.path.join(INTERMEDIATE_DATA_PATH, 'CEM.pkl')
KN_TRAIN_WIDS_CIDS_PKL = os.path.join(INTERMEDIATE_DATA_PATH, 'kn_train_wids_cids.pkl')
TEST_WIDS_CIDS_PKL = os.path.join(INTERMEDIATE_DATA_PATH, 'test_wids_cids.pkl')
KN_TRAIN_WIDS_CIDS_PADDED_PKL = os.path.join(INTERMEDIATE_DATA_PATH, 'kn_train_wids_cids_padded.pkl')
TEST_WIDS_CIDS_PADDED_PKL = os.path.join(INTERMEDIATE_DATA_PATH, 'test_wids_cids_padded.pkl')

WORDS_NUM = 20891
CHARS_NUM = 3048
WORD_EMBEDDING_DIM = 300
CHAR_EMBEDDING_DIM = 300

MAX_WSEQ_LEN = 39
MAX_CSEQ_LEN = 58

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