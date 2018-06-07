import os
import pandas as pd
import numpy as np
import glob
import shutil
import random
from collections import defaultdict

from sample_data_set import *

# 配置
RAW_DATA_SET_PATH = "/root/mounted/datasets/raw_data_0606/"
KN_PAIRS_CSV = os.path.join(RAW_DATA_SET_PATH, "knowledge_pairs.csv")
TRAIN_PAIRS_CSV = os.path.join(RAW_DATA_SET_PATH, "train_pairs.csv")
TEST_PAIRS_CSV = os.path.join(RAW_DATA_SET_PATH, 'test_all.csv')
NEED_COPY_PATH = os.path.join(RAW_DATA_SET_PATH, 'need_copy')
CHAR_EM_TXT = os.path.join(NEED_COPY_PATH, "char_embed.txt")
WORD_EM_TXT = os.path.join(NEED_COPY_PATH, "word_embed.txt")
QUESTION_CSV = os.path.join(NEED_COPY_PATH, "question.csv")

PUBLISH_PATH = "/root/mounted/datasets/publish_0606_tr50_test/"
PUBLISH_FRONT_END = os.path.join(PUBLISH_PATH, 'frontend')
PUBLISH_BACK_END = os.path.join(PUBLISH_PATH, 'backend')
PUBLISH_TRAIN_CSV = os.path.join(PUBLISH_FRONT_END, 'train.csv')
PUBLISH_TEST_CSV = os.path.join(PUBLISH_FRONT_END, 'test.csv')
PUBLISH_TEST_LABEL_CSV = os.path.join(PUBLISH_BACK_END, 'test_label.csv')

KN_TRAIN_SAMPLE_SEED = None
KN_TRAIN_SAMPLE_TARGET_RATE = 0.45
KN_TRAIN_SAMPLE_NUM = 200000

SPLIT_RATE = 0.25
SPLIT_TEST_TARGET_RATE = 0.5
SPLIT_SEED = None

SCORE_RATE = 0.375
SCORE_SEED = None

class KN_PAIRS(object):
    sep = '\t'
    columns = ['label', 'q1', 'q2']
    cmaps = {}


class TRAIN_PAIRS(object):
    sep = '\t'
    columns = ['label', 'q1', 'q2']
    cmaps = {}


class KN_TRAIN_PAIRS(object):
    sep = '\t'
    columns = ['label', 'q1', 'q2']
    cmaps = {}
    label = 'label'


class TEST_PAIRS(object):
    sep = '\t'
    columns = ['label', 'qid1', 'qid2']
    cmaps = {
        'qid1': 'q1',
        'qid2': 'q2'
    }


class KN_TRAIN_TEST_PAIRS(object):
    q1 = 'q1'
    q2 = 'q2'
    label = 'label'


class TEST(object):
    label = 'label'


def combine_kn_train(kn_pairs, train_pairs):
    # 重新排列列的顺序
    kn_pairs = kn_pairs.loc[:, KN_PAIRS.columns]
    train_pairs = train_pairs.loc[:, TRAIN_PAIRS.columns]

    # 统一kn和train的列名
    kn_pairs.rename(KN_PAIRS.cmaps, axis = 1, inplace=True)
    train_pairs.rename(TRAIN_PAIRS.cmaps, axis = 1, inplace=True)

    kn_train_pairs = pd.concat([kn_pairs, train_pairs], ignore_index=True)

    return kn_train_pairs


def combine_kn_train_test(kn_train_pairs, test_pairs):
    # 重新排列列的顺序
    kn_train_pairs = kn_train_pairs.loc[:, KN_TRAIN_PAIRS.columns]
    test_pairs = test_pairs.loc[:, TEST_PAIRS.columns]

    # 统一kn_train和test的列名
    kn_train_pairs.rename(KN_TRAIN_PAIRS.cmaps, axis = 1, inplace=True)
    test_pairs.rename(TEST_PAIRS.cmaps, axis = 1, inplace=True)

    kn_train_test_pairs = pd.concat([kn_train_pairs, test_pairs], ignore_index=True)

    return kn_train_test_pairs


def make_score_file(test_labels, pre_rate=0.5, random_state=None):
    """
    Args:
        test_labels: np.array, shape = (N, )
    """

    if random_state:
        np.random.seed(random_state)

    test_labels_len = len(test_labels)
    ones_num = int(max(0, test_labels_len * pre_rate))
    ones_num = min(ones_num, test_labels_len)
    zeros_num = test_labels_len - ones_num
    ones = np.ones(ones_num, dtype=np.int)
    zeros = np.zeros(zeros_num, dtype=np.int)
    ones_zeros = np.append(ones, zeros)
    ones_zeros_perm = np.random.permutation(ones_zeros)
    return pd.DataFrame(np.hstack([test_labels.reshape(-1, 1), ones_zeros_perm.reshape(-1, 1)]),
                        columns=['y_true', 'is_preliminary'])

def shuffle_q1_q2(data):
    q1 = data[KN_TRAIN_TEST_PAIRS.q1]
    q2 = data[KN_TRAIN_TEST_PAIRS.q2]
    q1_q2_zip = zip(q1, q2)

    def shuffle(q1_q2):
        q1, q2 = q1_q2
        return (q1, q2) if random.uniform(0,1) <=0.5 else (q2, q1)

    return pd.DataFrame(list(map(shuffle, q1_q2_zip)), columns=[KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2])


# 测试用##########################
def neighbours(data, q1, q2):
    q_dict = defaultdict(set)
    for i in range(data.shape[0]):
        q_dict[data[q1][i]].add(data[q2][i])
        q_dict[data[q2][i]].add(data[q1][i])

    return q_dict


def check_q1_q2_intersect(data, q1, q2, q_dict):

    def intersect(row):
        return(len(set(q_dict[row[q1]]).intersection(set(q_dict[row[q2]]))))

    return data.apply(intersect, axis = 1)
############################################



def main():
    # kn train整合
    print("正在整合kn与train...")
    kn_pairs = pd.read_csv(KN_PAIRS_CSV, sep=KN_PAIRS.sep)
    train_pairs = pd.read_csv(TRAIN_PAIRS_CSV, sep=TRAIN_PAIRS.sep)
    kn_train_pairs = combine_kn_train(kn_pairs, train_pairs)

    print("kn_train_pairs num: \n", len(kn_train_pairs))
    # kn_train采样
    print("正在对kn_train进行采样...")
    kn_train_pairs_sampled = sample_data_set(kn_train_pairs,
                                             KN_TRAIN_SAMPLE_TARGET_RATE,
                                             KN_TRAIN_SAMPLE_NUM,
                                             KN_TRAIN_SAMPLE_SEED,
                                             KN_TRAIN_PAIRS.label)

    print('kn_train_pairs_sampled num: \n', len(kn_train_pairs_sampled))
    # kn_train_sampled test整合
    print("正在整合kn_train_sampled与test...")
    test_pairs = pd.read_csv(TEST_PAIRS_CSV, sep = TEST_PAIRS.sep)
    kn_train_test_pairs = combine_kn_train_test(kn_train_pairs_sampled, test_pairs)

    print('kn_train_test_pairs: \n', len(kn_train_test_pairs))

    # 基于kn_train_test_sampled划分训练集和测试集 （shuffle?）
    print("正在从kn_train_test划分训练集和测试集...")
    train, test = train_test_split(kn_train_test_pairs,
                                   SPLIT_TEST_TARGET_RATE,
                                   SPLIT_RATE,
                                   SPLIT_SEED,
                                   True,
                                   KN_TRAIN_TEST_PAIRS.q1,
                                   KN_TRAIN_TEST_PAIRS.q2,
                                   KN_TRAIN_TEST_PAIRS.label)

    print("train num: \n", len(train))
    print("test num: \n", len(test))

    # train, test, 随机更换q1， q2的顺序
    print("更换train test中q1和q2的顺序")
    train[[KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2]] = shuffle_q1_q2(train)
    test[[KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2]] = shuffle_q1_q2(test)

    # # 测试用 ########################
    # print("丢弃训练集中intersecttion大于测试集的样本")
    # train_test = pd.concat([train, test], ignore_index=True)
    # qn_dict = neighbours(train_test, KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2)
    # train[Q1_Q2_INTERSECT] = check_q1_q2_intersect(train, KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2, qn_dict)
    # test[Q1_Q2_INTERSECT] = check_q1_q2_intersect(test, KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2, qn_dict)
    # test_inter_max = test[Q1_Q2_INTERSECT].max()
    # train = train[train[Q1_Q2_INTERSECT] <= test_inter_max]
    # train.drop(labels = Q1_Q2_INTERSECT, axis = 1, inplace = True)
    # test.drop(labels = Q1_Q2_INTERSECT, axis = 1, inplace = True)
    # print("train num after drop: \n", len(train))
    # print("test num: \n", len(test))
    # # ###############################

    if not os.path.exists(PUBLISH_PATH):
        os.mkdir(PUBLISH_PATH)

    if not os.path.exists(PUBLISH_FRONT_END):
        os.mkdir(PUBLISH_FRONT_END)

    if not os.path.exists(PUBLISH_BACK_END):
        os.mkdir(PUBLISH_BACK_END)

    # 保存训练集
    print("正在保存训练集...")
    train.to_csv(PUBLISH_TRAIN_CSV, index=False)

    # 生成测试集打分文件
    print("正在生成测试集打分文件...")
    test_label = make_score_file(test[TEST.label].values, SCORE_RATE, SCORE_SEED)
    test_label.to_csv(PUBLISH_TEST_LABEL_CSV, index=False)

    # 保存去掉label的测试集
    print("正在生成测试集...")
    test.drop(TEST.label, axis = 1, inplace = True)
    test.to_csv(PUBLISH_TEST_CSV, index=False)

    # 拷贝其他数据
    print("正在拷贝其他相关数据...")
    files = glob.glob(os.path.join(NEED_COPY_PATH, "*"))
    for f in files:
        print(f)
        _, fn = os.path.split(f)
        shutil.copyfile(f, os.path.join(PUBLISH_FRONT_END, fn))

    print("Done")


if __name__ == '__main__':
    main()