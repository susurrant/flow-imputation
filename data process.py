# -*- coding: utf-8 -*-：

import numpy as np
import csv
import pysal as ps
import random

# 出租车数据处理
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


# 自然间断点
def fisher_jenks(d, cNum):
    X = np.array(d).reshape((-1, 1))
    fj = ps.esda.mapclassify.Fisher_Jenks(X, cNum)
    meanV = []
    for i in range(cNum):
        meanV.append(np.mean(X[np.where(i == fj.yb)]))
    return fj.bins, meanV


# 数据分级
def classification(filename, class_num, threshold):
    data = []
    flows = {}
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            d = line.strip().split('\t')
            if d[0] == d[1] or int(d[-1]) < threshold: # 不考虑自身到自身 且 交互强度大于等于阈值
                line = f.readline()
                continue
            data.append(int(d[-1]))
            flows[(d[0], d[1])] = int(d[-1])
            line = f.readline()
    nk, nl = fisher_jenks(data, class_num)
    print(nk, nl)

    with open(filename[:-4]+'_c'+str(class_num)+'_t'+str(threshold)+filename[-4:], 'w', newline='') as rf:
        sheet = csv.writer(rf, delimiter='\t')
        sheet.writerow(['ogrid', 'relation', 'dgrid', 'm'])
        for g, m in flows.items():
            x = np.where(m <= nk)[0]
            i = x.min() if x.size > 0 else len(nk) - 1
            sheet.writerow([g[0], i, g[1], m])


def read_features(entity_dict, feature_file):
    grid_list = list(np.loadtxt(entity_dict, dtype=np.uint16, delimiter='\t')[:, 1])
    features = {}
    with open(feature_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sl = line.strip().split('\t')
            features[grid_list[i]] = list(map(int, sl[2:]))  #[attract, pull]
    return features


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


def gen_data(flow_file, output_path, r):
    p_data = np.loadtxt(flow_file, dtype=np.uint16, delimiter='\t', skiprows=1)
    grids = list(set(p_data[:, 0]) | set(p_data[:, 2]))

    # 生成训练、测试、验证数据集
    # positive triplets
    t = set(range(p_data.shape[0]))
    train_set = set(random.sample(range(p_data.shape[0]), int(r[0] * p_data.shape[0])))
    train_data = p_data[list(train_set)]
    s = t - train_set
    test_set = set(random.sample(s, int(r[1] * p_data.shape[0])))
    test_data = p_data[list(test_set)]
    valid_data = p_data[list(s - test_set)]

    np.random.shuffle(train_data)
    np.savetxt(output_path + 'train.txt', train_data, fmt='%d', delimiter='\t')
    np.savetxt(output_path + 'test.txt', test_data, fmt='%d', delimiter='\t')
    np.savetxt(output_path + 'valid.txt', valid_data, fmt='%d', delimiter='\t')

    # 生成数据字典
    with open(output_path + 'entities.dict', 'w', newline='') as f:
        for i, gid in enumerate(grids):
            f.write(str(i)+'\t'+ str(gid) + '\r\n')

    with open(output_path + 'relations.dict', 'w', newline='') as f:
        relations = set(p_data[:, 1])
        for i, r in enumerate(relations):
            f.write(str(i) + '\t' + str(r) + '\r\n')


# 生成节点特征
def gen_features(flow_file, output_path, colnum, normalizaed=False):
    features = [] # [row, col, attract, pull]
    node_list = []
    with open(output_path + 'entities.dict', 'r') as f:
        line = f.readline().strip()
        while line:
            s = line.split('\t')
            features.append([int(s[1])//colnum, int(s[1])%colnum, 0, 0]) # 0, 0
            node_list.append(s[1])
            line = f.readline().strip()

    with open(flow_file, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            s = line.split('\t')
            features[node_list.index(s[0])][3] += int(s[-1]) # 3    # pick-up:  pull
            features[node_list.index(s[2])][2] += int(s[-1])        # drop-off: attract
            line = f.readline().strip()

    features = np.array(features, dtype=np.float)

    np.savetxt(output_path + 'features_raw.txt', features, fmt='%d', delimiter='\t')
    if normalizaed:
        features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

    np.savetxt(output_path + 'features.txt', features, fmt='%.3f', delimiter='\t')


def break_calc(filename, c_num):
    data = []
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            d = line.split(',')
            data.append(int(d[-1]))
            line = f.readline().strip()
    print(max(data))
    nl, nk = fisher_jenks(data, c_num)
    print(nl)
    print(nk)


if __name__ == '__main__':
    #taxi_data('data/taxi_sj_1km_05.txt', 'data/taxi_1km.txt')
    class_num = 1
    threshold = 30
    col_num = 30
    path = 'SI-GCN/data/taxi/'

    #classification('data/taxi_1km.txt', class_num, threshold)
    flow_file = 'data/taxi_1km_c'+str(class_num)+'_t'+str(threshold)+'.txt'
    gen_data(flow_file, path, [0.6, 0.2, 0.2])
    gen_features(flow_file, path, colnum=col_num, normalizaed=True)
    #sample_negatives(flow_file, path)

