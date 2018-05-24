import numpy as np
from collections import defaultdict
import pandas as pd

Q1_Q2_INTERSECT = 'q1_q2_intersect'

def sample(df, num):   
    shuffled_indice = np.random.permutation(df.shape[0])
    return df.iloc[shuffled_indice[:num],:].copy(), df.iloc[shuffled_indice[num:],:].copy()


def sample_data_set(all_pairs, target_rate, sampling_number, random_state = None, label = 'label'):
    if random_state:
        np.random.seed(random_state)
        
    pos_num = int(sampling_number * target_rate)
    neg_num = sampling_number - pos_num
    
    pos_pairs = all_pairs[all_pairs[label] == 1]
    neg_pairs = all_pairs[all_pairs[label] == 0]
    
    pos_sampled = sample(pos_pairs, pos_num)[0]
    neg_sampled = sample(neg_pairs, neg_num)[0]
    return pd.concat([pos_sampled, neg_sampled], ignore_index=True)


def q1_q2_intersect(data, q1='q1', q2='q2'):
    q_dict = defaultdict(set)
    for i in range(data.shape[0]):
        q_dict[data[q1][i]].add(data[q2][i])
        q_dict[data[q1][i]].add(data[q2][i])
    
    def intersect(row):
        return(len(set(q_dict[row[q1]]).intersection(set(q_dict[row[q2]]))))
    
    return data.apply(intersect, axis = 1)


def split_test_train_set(data_set, target_rate, split_rate = 0.2, label = 'label'):
    inter_nums = set(data_set[Q1_Q2_INTERSECT])
    sampled_test = []
    sampled_train = []
    for i in inter_nums:
        data = data_set[data_set[Q1_Q2_INTERSECT] == i]
        pos_pairs = data[data['label'] == 1]
        neg_pairs = data[data['label'] == 0]
        
        pos_split_num = int(data.shape[0] * split_rate * target_rate)
        neg_split_num = int(data.shape[0] * split_rate * (1 - target_rate))
        
        if pos_split_num <= pos_pairs.shape[0] and neg_split_num <= neg_pairs.shape[0]:
            pos_pairs_sampled_test, pos_pairs_sampled_train = sample(pos_pairs, pos_split_num)
            neg_pairs_sampled_test, neg_pairs_sampled_train = sample(neg_pairs, neg_split_num)
        
        if pos_split_num > pos_pairs.shape[0]:
            # pos_sample_num = int(pos_pairs.shape[0] * split_rate)
            pos_sample_num = int(pos_pairs.shape[0] * 0.5)
            neg_sample_num = int(pos_sample_num * (1 / target_rate - 1))
            pos_pairs_sampled_test, pos_pairs_sampled_train = sample(pos_pairs, pos_sample_num)
            neg_pairs_sampled_test, neg_pairs_sampled_train = sample(neg_pairs, neg_sample_num)
        
        if neg_split_num > neg_pairs.shape[0]:
            # neg_sample_num = int(neg_pairs.shape[0] * split_rate)
            neg_sample_num = int(neg_pairs.shape[0] *0.5)
            pos_sample_num = int(neg_sample_num * (target_rate / (1 - target_rate)))
            pos_pairs_sampled_test, pos_pairs_sampled_train = sample(pos_pairs, pos_sample_num)
            neg_pairs_sampled_test, neg_pairs_sampled_train = sample(neg_pairs, neg_sample_num)            
        
        sampled_test.append(pd.concat([pos_pairs_sampled_test, neg_pairs_sampled_test],ignore_index=True))
        sampled_train.append(pd.concat([pos_pairs_sampled_train, neg_pairs_sampled_train], ignore_index=True))
        
    return pd.concat(sampled_test, ignore_index=True), pd.concat(sampled_train, ignore_index=True)


def train_test_split(data_set, test_target_rate, split_rate = 0.2, random_state = None, drop_intersect = False, q1 = 'q1', q2 = 'q2', label = 'label'):
    if random_state:
        np.random.seed(random_state)
        
    data_set_copy = data_set.copy()
    data_set_copy[Q1_Q2_INTERSECT] = q1_q2_intersect(data_set_copy, q1, q2)
    sampled_test, sampled_train = split_test_train_set(data_set_copy, test_target_rate, split_rate, label)
    
    if drop_intersect:
        sampled_train.drop(Q1_Q2_INTERSECT, axis = 1, inplace = True)
        sampled_test.drop(Q1_Q2_INTERSECT, axis = 1, inplace = True)
        
    return sampled_train, sampled_test