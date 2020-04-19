# -*- coding: utf-8 -*-：

import numpy as np
import csv
import random

# taxi data process
def taxi_data(data_file, out_file):
    flows = {}
    flow_count = 0
    with open(data_file, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            d1 = line.split(',')
            d2 = f.readline().strip().split(',')
            if d1[1] == '1' and d2[1] == '1':
                k = (d1[-1], d2[-1])
                if k not in flows:
                    flows[k] = 0
                flows[k] += 1
                flow_count += 1
            line = f.readline().strip()

    print('number of taxi trips:', flow_count)

    with open(out_file, 'w', newline='') as rf:
        sheet = csv.writer(rf, delimiter='\t')
        sheet.writerow(['ogid', 'dgid', 'm'])
        for g, m in flows.items():
            sheet.writerow([g[0], g[1], m])


# data classfication
def data_filter(filename, threshold):
    with open(filename[:-4] + '_t' + str(threshold) + filename[-4:], 'w', newline='') as rf:
        sheet = csv.writer(rf, delimiter='\t')
        sheet.writerow(['ogrid', 'relation', 'dgrid', 'm'])
        with open(filename, 'r') as f:
            f.readline()
            line = f.readline()
            while line:
                d = line.strip().split('\t')
                if d[0] == d[1] or int(d[-1]) < threshold: # 不考虑自身到自身 且 交互强度大于等于阈值
                    line = f.readline()
                    continue
                sheet.writerow([d[0], 0, d[1], d[-1]])
                line = f.readline()


def read_features(entity_dict, feature_file):
    grid_list = list(np.loadtxt(entity_dict, dtype=np.uint16, delimiter='\t')[:, 1])
    features = {}
    with open(feature_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sl = line.strip().split('\t')
            features[grid_list[i]] = list(map(int, sl[2:]))  #[attract, pull]
    return features


'''
# generate negative samples to test_n.txt
def sample_negatives_weighted(flow_file, output_path):
    p_data = np.loadtxt(flow_file, dtype=np.uint16, delimiter='\t', skiprows=1)
    grids = list(set(p_data[:, 0]) | set(p_data[:, 2]))
    sample_size = 0
    with open(output_path+'test.txt', 'r') as f:
        for _ in f:
            sample_size += 1
    features = read_features(output_path+'entities.dict', output_path+'features_raw.txt') #[attract, pull]
    p_dict = set(map(tuple, p_data[:, [0, 2]]))
    n_dict = set()
    negatives = []
    for i in range(len(grids)):
        for j in range(len(grids)):
            if j != i:
                n_dict.add((grids[i], grids[j]))
    n_dict -= p_dict
    for g in n_dict:
        negatives.append([g[0], 0, g[1], 0, features[g[0]][1] + features[g[1]][0]])  # 只考虑一种关系，第二项值为0
    negatives = np.array(negatives)
    np.random.shuffle(negatives)
    n_idx = np.random.choice(negatives.shape[0], size=sample_size, replace=False, p=negatives[:, 4] / np.sum(negatives[:, 4]))
    np.savetxt(output_path + 'test_n.txt', negatives[n_idx][:, :4], fmt='%d', delimiter='\t')


def sample_negatives(flow_file, output_path):
    p_data = np.loadtxt(flow_file, dtype=np.uint16, delimiter='\t', skiprows=1)
    grids = list(set(p_data[:, 0]) | set(p_data[:, 2]))
    sample_size = 0
    with open(output_path+'test.txt', 'r') as f:
        for _ in f:
            sample_size += 1
    p_dict = set(map(tuple, p_data[:, [0, 2]]))
    n_dict = set()
    negatives = []
    for i in range(len(grids)):
        for j in range(len(grids)):
            if j != i:
                n_dict.add((grids[i], grids[j]))
    n_dict -= p_dict
    for g in n_dict:
        negatives.append([g[0], 0, g[1], 0])  # 只考虑一种关系，第二项值为0
    negatives = np.array(negatives)
    np.random.shuffle(negatives)
    n_idx = np.random.choice(negatives.shape[0], size=sample_size, replace=False)
    np.savetxt(output_path + 'test_n.txt', negatives[n_idx], fmt='%d', delimiter='\t')
'''

def gen_data(flow_file, output_path, r, mode = 'random'):
    p_data = np.loadtxt(flow_file, dtype=np.uint16, delimiter='\t', skiprows=1)
    grids = list(set(p_data[:, 0]) | set(p_data[:, 2]))
    t = set(range(p_data.shape[0]))

    # generate training, validation and test data set (positive flows)
    train_size = int(r[0] * p_data.shape[0])
    if mode == 'random':
        weights = np.ones(len(p_data))
    elif mode == 'high weight':
        weights = p_data[:, 3]
    elif mode == 'low weight':
        weights = 1 / p_data[:, 3]

    weights = weights/np.sum(weights)
    train_set = set(np.random.choice(p_data.shape[0], size=train_size, replace=False, p=weights))

    s = t - train_set
    test_set = set(random.sample(s, int(r[1] * p_data.shape[0])))

    train_data = p_data[list(train_set)]
    test_data = p_data[list(test_set)]
    valid_data = p_data[list(s - test_set)]

    np.random.shuffle(train_data)
    np.savetxt(output_path + 'train.txt', train_data, fmt='%d', delimiter='\t')
    np.savetxt(output_path + 'test.txt', test_data, fmt='%d', delimiter='\t')
    np.savetxt(output_path + 'valid.txt', valid_data, fmt='%d', delimiter='\t')

    # generate geographical unit and spatial relation dicts
    with open(output_path + 'entities.dict', 'w', newline='') as f:
        for i, gid in enumerate(grids):
            f.write(str(i)+'\t'+ str(gid) + '\r\n')

    with open(output_path + 'relations.dict', 'w', newline='') as f:
        relations = set(p_data[:, 1])
        for i, r in enumerate(relations):
            f.write(str(i) + '\t' + str(r) + '\r\n')


# generate the features of geographical units
def gen_features(flow_file, output_path, colnum, mode='entire'):
    features = [] # [row, col, push, pull]
    node_list = []
    with open(output_path + 'entities.dict', 'r') as f:
        line = f.readline().strip()
        while line:
            s = line.split('\t')
            if mode == 'entire':
                features.append([int(s[1])//colnum, int(s[1])%colnum, 0, 0]) # co, push, pull
            elif mode == 'limited':
                features.append([int(s[1]) // colnum, int(s[1]) % colnum])   # include only coordinates
            else:
                features.append([mode])                                         # specific value
            node_list.append(s[1])
            line = f.readline().strip()

    if mode == 'entire':
        with open(flow_file, 'r') as f:
            f.readline()
            line = f.readline().strip()
            while line:
                s = line.split('\t')
                features[node_list.index(s[0])][3] += int(s[-1])        # pick-up:  push
                features[node_list.index(s[2])][2] += int(s[-1])        # drop-off: pull
                line = f.readline().strip()

    features = np.array(features, dtype=np.float)

    np.savetxt(output_path + 'features_raw.txt', features, fmt='%d', delimiter='\t')

    # save normalized features
    if mode != 'none':
        features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    np.savetxt(output_path + 'features.txt', features, fmt='%.3f', delimiter='\t')



if __name__ == '__main__':
    # 1500m, 20; 1km 30; 500m, 59

    scale = '500m'
    col_num = 59

    threshold = 30
    tvt = [0.6, 0.2, 0.2]

    #taxi_data('data/sj_taxi_'+scale+'_051317.txt', 'data/taxi_'+scale+'.txt')
    path = 'SI-GCN/data/taxi_'+scale+'_th'+str(threshold)+'/'

    data_filter('data/taxi_'+scale+'.txt', threshold)
    flow_file = 'data/taxi_'+scale+'_t'+str(threshold)+'.txt'
    gen_data(flow_file, path, tvt, mode='random') # random, low weight, hight weight
    gen_features(flow_file, path, colnum=col_num, mode='entire') # entire, limited, specific value (e.g., 1)