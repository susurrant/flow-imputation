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
        sheet.writerow(['osid', 'dsid', 'm'])
        for g, m in flows.items():
            sheet.writerow([g[0], g[1], m])


# data classfication
def data_filter(filename, threshold):
    with open(filename[:-4] + '_t' + str(threshold) + filename[-4:], 'w', newline='') as rf:
        sheet = csv.writer(rf, delimiter='\t')
        sheet.writerow(['osid', 'relation', 'dsid', 'm'])
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


def gen_data(flow_file, output_path, r, mode = 'random'):
    p_data = np.loadtxt(flow_file, dtype=np.uint16, delimiter='\t', skiprows=1)
    streets = list(set(p_data[:, 0]) | set(p_data[:, 2]))
    t = set(range(p_data.shape[0]))

    # generate training, validation and test data set (positive flows)
    train_size = int(r[0] * p_data.shape[0])
    if mode == 'random':
        weights = np.ones(len(p_data))
    elif mode == 'high weight':
        weights = p_data[:, 3]
    elif mode == 'low weight':
        weights = 1/p_data[:, 3]

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
        for i, sid in enumerate(streets):
            f.write(str(i)+'\t'+ str(sid) + '\r\n')

    with open(output_path + 'relations.dict', 'w', newline='') as f:
        relations = set(p_data[:, 1])
        for i, r in enumerate(relations):
            f.write(str(i) + '\t' + str(r) + '\r\n')


# generate the features of geographical units
def gen_features(flow_file, street_file, output_path):
    street_dict = {}
    with open(street_file, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            s = line.split(',')
            street_dict[s[1]] = (float(s[2]), float(s[3]))
            line = f.readline().strip()


    features = [] # [x, y, attract, pull]
    node_list = []
    with open(output_path + 'entities.dict', 'r') as f:
        line = f.readline().strip()
        while line:
            s = line.split('\t')
            features.append([street_dict[s[1]][0], street_dict[s[1]][1], 0, 0]) # 0, 0
            node_list.append(s[1])
            line = f.readline().strip()

    with open(flow_file, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            s = line.split('\t')
            features[node_list.index(s[0])][3] += int(s[-1])        # pick-up:  pull
            features[node_list.index(s[2])][2] += int(s[-1])        # drop-off: attract
            line = f.readline().strip()

    features = np.array(features, dtype=np.float)

    np.savetxt(output_path + 'features_raw.txt', features, fmt='%.3f', delimiter='\t')

    # save normalized features
    features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    np.savetxt(output_path + 'features.txt', features, fmt='%.3f', delimiter='\t')


if __name__ == '__main__':
    path = 'SI-GCN/data/taxi_roadseg/'

    # ------------- step 1 -------------
    #taxi_data('data/sj_taxi_roadseg_051317.txt', 'data/taxi_roadseg.txt')

    # ------------- step 2 -------------
    threshold = 30
    data_filter('data/taxi_roadseg.txt', threshold)

    # ------------- step 3 -------------
    flow_file = 'data/taxi_roadseg'+'_t'+str(threshold)+'.txt'
    street_file = 'data/pt_roadseg_5huan.txt'
    gen_data(flow_file, path, [0.6, 0.2, 0.2], mode='random') # random, low weight, high weight
    gen_features(flow_file, street_file, path)
