from config import *
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def add_wv_to_qpairs(test_df, qs, w2i, c2i):
    
    def words2ids(item):
        return np.array(list(map(lambda x: w2i[x],item.split())))

    def chars2ids(item):
        return np.array(list(map(lambda x: c2i[x],item.split())))
    
    def pad_wseq(item):
        return pad_sequences(item, maxlen=MAX_WSEQ_LEN)

    def pad_cseq(item):
        return pad_sequences(item, maxlen=MAX_CSEQ_LEN)
    
    test_wv_cv = pd.merge(test_df, qs, left_on='qid1', right_on='qid')
    test_wv_cv.drop(labels='qid', axis=1, inplace=True)
    test_wv_cv = pd.merge(test_wv_cv, qs, left_on='qid2', right_on='qid', suffixes=('1', '2'))
    test_wv_cv.drop(labels='qid', axis=1, inplace=True)
    
    test_wv_cv_words1, test_wv_cv_words2 = test_wv_cv['words1'], test_wv_cv['words2']
    test_wv_cv_chars1, test_wv_cv_chars2 = test_wv_cv['chars1'], test_wv_cv['chars2']
    
    test_wids_cids = test_wv_cv.copy()
    
    test_wids_cids['words1'] = test_wv_cv_words1.map(words2ids)
    test_wids_cids['words2'] = test_wv_cv_words2.map(words2ids)
    test_wids_cids['chars1'] = test_wv_cv_chars1.map(chars2ids)
    test_wids_cids['chars2'] = test_wv_cv_chars2.map(chars2ids)
    
    
    test_wids_cids_words1, test_wids_cids_words2 = test_wids_cids['words1'], test_wids_cids['words2']
    test_wids_cids_chars1, test_wids_cids_chars2 = test_wids_cids['chars1'], test_wids_cids['chars2']
    
    test_wids_cids_copy = test_wids_cids.copy()
    
    test_wids_cids_copy['words1'] = list(pad_sequences(test_wids_cids_words1, maxlen=MAX_WSEQ_LEN)) # kn_train_wids_cids_words1.map(pad_wseq)
    test_wids_cids_copy['words2'] = list(pad_sequences(test_wids_cids_words2, maxlen=MAX_WSEQ_LEN)) # kn_train_wids_cids_words2.map(pad_wseq)
    test_wids_cids_copy['chars1'] = list(pad_sequences(test_wids_cids_chars1, maxlen=MAX_CSEQ_LEN)) # kn_train_wids_cids_chars1.map(pad_cseq)
    test_wids_cids_copy['chars2'] = list(pad_sequences(test_wids_cids_chars2, maxlen=MAX_CSEQ_LEN)) # kn_train_wids_cids_chars2.map(pad_cseq)
    
    return test_wids_cids_copy