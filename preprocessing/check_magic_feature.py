import os
import pandas as pd
from collections import Counter, defaultdict

from publish_data import *

def freq(train_test, q1, q2):
    qs1 = train_test[q1]
    qs2 = train_test[q2]
    qs = qs1.append(qs2, ignore_index=True)
    qs_counter = Counter(qs)

    return qs_counter


def q_freq(data, q, qs_counter):
    return data.apply(lambda row: qs_counter[row[q]], axis = 1)


def neighbours(data, q1, q2):
    q_dict = defaultdict(set)
    for i in range(data.shape[0]):
        q_dict[data[q1][i]].add(data[q2][i])
        q_dict[data[q1][i]].add(data[q2][i])

    return q_dict


def q1_q2_intersect(data, q1, q2, q_dict):

    def intersect(row):
        return(len(set(q_dict[row[q1]]).intersection(set(q_dict[row[q2]]))))

    return data.apply(intersect, axis = 1)


def main():
    # 加载训练集、测试集、打分文件
    print("读取train.csv, test.csv, test_label.csv...")
    train = pd.read_csv(PUBLISH_TRAIN_CSV)
    test = pd.read_csv(PUBLISH_TEST_CSV)
    test_label = pd.read_csv(PUBLISH_TEST_LABEL_CSV)

    # 把测试集和打分文件结合到一起
    print("合并test_label和test...")
    test[TEST.label] = test_label.y_true
    test = test[KN_TRAIN_PAIRS.columns]

    # 训练集和测试集concat到一起，统计各个问题出现的数目，intersect
    print("统计freq和neighbour...")
    train_test = pd.concat([train, test], ignore_index=True)
    qs_counter = freq(train_test, KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2)
    qn_dict = neighbours(train_test, KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2)

    # 训练集和测试集分别添加freq列和intersect列
    print("分别给训练集和测试集添加q1_freq和q2_freq...")
    train['q1_freq'] = q_freq(train, KN_TRAIN_TEST_PAIRS.q1, qs_counter)
    train['q2_freq'] = q_freq(train, KN_TRAIN_TEST_PAIRS.q2, qs_counter)
    test['q1_freq'] = q_freq(test, KN_TRAIN_TEST_PAIRS.q1, qs_counter)
    test['q2_freq'] = q_freq(test, KN_TRAIN_TEST_PAIRS.q2, qs_counter)
    print("分别给训练集和测试集添加Q1_Q2_INTERSECT...")
    train[Q1_Q2_INTERSECT] = q1_q2_intersect(train, KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2, qn_dict)
    test[Q1_Q2_INTERSECT] = q1_q2_intersect(test, KN_TRAIN_TEST_PAIRS.q1, KN_TRAIN_TEST_PAIRS.q2, qn_dict)

    # 分别计算两个magicfeature在训练集和测试集的相关性
    print(train.head())
    print(test.head())
    print(train.corr())
    print(test.corr())

if __name__ == '__main__':
    main()