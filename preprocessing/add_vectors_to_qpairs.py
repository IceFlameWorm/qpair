from config import *
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

def add_wv_to_qpairs(test_df, qs, w2i, c2i, df_len):
    
    def words2ids(item):
        return np.array(list(map(lambda x: w2i[x],item.split())))

    def chars2ids(item):
        return np.array(list(map(lambda x: c2i[x],item.split())))
    
    def pad_wseq(item):
        return pad_sequences(item, maxlen=MAX_WSEQ_LEN)

    def pad_cseq(item):
        return pad_sequences(item, maxlen=MAX_CSEQ_LEN)
    
#     test_wv_cv = pd.merge_ordered(test_df, qs, left_on='qid1', right_on='qid')
#     test_wv_cv.drop(labels='qid', axis=1, inplace=True)
#     test_wv_cv = pd.merge_ordered(test_wv_cv, qs, left_on='qid2', right_on='qid', suffixes=('1', '2'))
#     test_wv_cv.drop(labels='qid', axis=1, inplace=True)
    
#     test_wv_cv_words1, test_wv_cv_words2 = test_wv_cv['words1'], test_wv_cv['words2']
#     test_wv_cv_chars1, test_wv_cv_chars2 = test_wv_cv['chars1'], test_wv_cv['chars2']
    
#     test_wids_cids = test_wv_cv.copy()
    
#     test_wids_cids['words1'] = test_wv_cv_words1.map(words2ids)
#     test_wids_cids['words2'] = test_wv_cv_words2.map(words2ids)
#     test_wids_cids['chars1'] = test_wv_cv_chars1.map(chars2ids)
#     test_wids_cids['chars2'] = test_wv_cv_chars2.map(chars2ids)
    
    
#     test_wids_cids_words1, test_wids_cids_words2 = test_wids_cids['words1'], test_wids_cids['words2']
#     test_wids_cids_chars1, test_wids_cids_chars2 = test_wids_cids['chars1'], test_wids_cids['chars2']
    
#     test_wids_cids_copy = test_wids_cids.copy()
    
#     test_wids_cids_copy['words1'] = list(pad_sequences(test_wids_cids_words1, maxlen=MAX_WSEQ_LEN)) # kn_train_wids_cids_words1.map(pad_wseq)
#     test_wids_cids_copy['words2'] = list(pad_sequences(test_wids_cids_words2, maxlen=MAX_WSEQ_LEN)) # kn_train_wids_cids_words2.map(pad_wseq)
#     test_wids_cids_copy['chars1'] = list(pad_sequences(test_wids_cids_chars1, maxlen=MAX_CSEQ_LEN)) # kn_train_wids_cids_chars1.map(pad_cseq)
#     test_wids_cids_copy['chars2'] = list(pad_sequences(test_wids_cids_chars2, maxlen=MAX_CSEQ_LEN)) # kn_train_wids_cids_chars2.map(pad_cseq)
    
    qs_copy = qs.copy()
    qs_copy_words = qs_copy['words']
    qs_copy_chars = qs_copy['chars']
    qs_copy['wids'] = qs_copy_words.map(words2ids)
    qs_copy['cids'] = qs_copy_chars.map(chars2ids)
    
    qid2wids = dict(zip(qs_copy['qid'], qs_copy['wids']))
    qid2cids = dict(zip(qs_copy['qid'], qs_copy['cids']))
    
    pbar = tqdm(total = df_len)
    
    def test_apply(_row):
        row = _row.copy()
        qid1, qid2 = row['qid1'], row['qid2']
        wids1 = qid2wids[qid1]
        wids2 = qid2wids[qid2]
        cids1 = qid2cids[qid1]
        cids2 = qid2cids[qid2]
        
        row['words1'] = list(pad_sequences([wids1], maxlen=MAX_WSEQ_LEN))[0]
        row['words2'] = list(pad_sequences([wids2], maxlen=MAX_WSEQ_LEN))[0]
        row['chars1'] = list(pad_sequences([cids1], maxlen=MAX_WSEQ_LEN))[0]
        row['chars2'] = list(pad_sequences([cids2], maxlen=MAX_WSEQ_LEN))[0]
        
        pbar.update(1)
        
        return row
        
    test_wids_cids_copy = test_df.apply(test_apply, axis = 1)
    pbar.close()
    
    return test_wids_cids_copy