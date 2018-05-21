import os

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