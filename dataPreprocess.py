# -*- coding: utf-8 -*-：

import numpy as np
import csv
import pysal as ps
import matplotlib.pyplot as plt
import random


# 联通数据处理
def unicom_data():
    grid_map = {}
    with open('data/250_is_1km.csv', 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            d = line.strip().split(',')
            grid_map[d[1]] = d[2] # map 250 to 500/1km
            line = f.readline()

    flows_250 = {}
    with open('data/unicom_OD.csv', 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            d = line.strip().split(',')
            if d[0] in ['20170109', '20170110', '20170111', '20170112', '20170113']:
                if (d[2], d[3]) not in flows_250:
                    flows_250[(d[2], d[3])] = 0
                flows_250[(d[2], d[3])] += int(d[4])
            line = f.readline()

    flows_sta = {}
    for g, m in flows_250.items():
        if g[0] in grid_map and g[1] in grid_map:
            k = (grid_map[g[0]], grid_map[g[1]])
            if k not in flows_sta:
                flows_sta[k] = 0
            flows_sta[k] += m

    with open('data/unicom_1km.csv', 'w', newline='') as rf:
        sheet = csv.writer(rf)
        sheet.writerow(['o', 'd', 'm'])
        for g, m in flows_sta.items():
            sheet.writerow([g[0], g[1], m])


# 出租车数据处理
def taxi_data():
    flows = {}
    with open('data/taxi_sj_1km_051317.csv', 'r') as f:
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
            line = f.readline().strip()

    with open('data/taxi_1km.csv', 'w', newline='') as rf:
        sheet = csv.writer(rf)
        sheet.writerow(['o', 'd', 'm'])
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
            d = line.strip().split(',')
            if d[0] == d[1] or int(d[-1]) < threshold: # 不考虑自身到自身 且 交互强度大于阈值
                line = f.readline()
                continue
            data.append(int(d[-1]))
            flows[(d[0], d[1])] = int(d[-1])
            line = f.readline()
    nk, nl = fisher_jenks(data, class_num)
    print(nk, nl)

    with open(filename[:-4]+'_c'+str(class_num)+'_t'+str(threshold)+filename[-4:], 'w', newline='') as rf:
        sheet = csv.writer(rf)
        sheet.writerow(['o', 'd', 'm', 'c'])
        for g, m in flows.items():
            x = np.where(m <= nk)[0]
            i = x.min() if x.size > 0 else len(nk) - 1
            sheet.writerow([g[0], g[1], m, i])


def gen_data(data_file, r, output_path):
    data = np.loadtxt(data_file, dtype=np.uint16, delimiter=',', skiprows=1)

    # 生成训练、测试、验证数据集
    t = set(range(data.shape[0]))
    train_set = set(random.sample(range(data.shape[0]), int(r[0] * data.shape[0])))
    s = t - train_set
    test_set = set(random.sample(s, int(r[1] * data.shape[0])))
    valid_set = s - test_set

    with open(output_path + 'train.txt', 'w', newline='') as f:
        for i in train_set:
            f.write(str(data[i, 0]) + '\t' + str(data[i, 3]) + '\t' + str(data[i, 1]) + '\r\n')

    with open(output_path + 'test.txt', 'w', newline='') as f:
        for i in test_set:
            f.write(str(data[i, 0]) + '\t' + str(data[i, 3]) + '\t' + str(data[i, 1]) + '\r\n')

    with open(output_path + 'valid.txt', 'w', newline='') as f:
        for i in valid_set:
            f.write(str(data[i, 0]) + '\t' + str(data[i, 3]) + '\t' + str(data[i, 1]) + '\r\n')

    # 生成数据字典
    with open(output_path + 'entities.dict', 'w', newline='') as f:
        grids = set(data[:, 0]) | set(data[:, 1])
        for i, gid in enumerate(grids):
            f.write(str(i)+'\t'+ str(gid) + '\r\n')

    with open(output_path + 'relations.dict', 'w', newline='') as f:
        relations = set(data[:, 3])
        for i, r in enumerate(relations):
            f.write(str(i) + '\t' + str(r) + '\r\n')


# 生成节点特征
def gen_features(entity_file, flow_file, output_file, colnum, normalizaed=False):
    features = [] # [row, col, attract, pull]
    node_list = []
    with open(entity_file, 'r') as f:
        line = f.readline().strip()
        while line:
            s = line.split('\t')
            features.append([int(s[1])//colnum, int(s[1])%colnum, 0, 0])
            node_list.append(s[1])
            line = f.readline().strip()

    with open(flow_file, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            s = line.split(',')
            features[node_list.index(s[0])][3] += int(s[2])
            features[node_list.index(s[1])][2] += int(s[2])
            line = f.readline().strip()

    features = np.array(features, dtype=np.float)

    if normalizaed:
        features /= np.max(features, axis=0)

    np.savetxt(output_file, features, fmt='%.3f', delimiter='\t')


if __name__ == '__main__':
    #unicom_data()
    #taxi_data()
    classification('data/taxi_1km.csv', 3, 50)
    c_file = 'data/taxi_1km_c3_t50.csv'
    path = 'SR-GCN/data/taxi_c3/'
    gen_data(c_file, [0.6, 0.2, 0.2], path)
    gen_features(path+'entities.dict', c_file, path+'features.txt', colnum=25, normalizaed=True)