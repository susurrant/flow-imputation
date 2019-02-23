# -*- coding: utf-8 -*-：

import numpy as np
import csv
import pysal as ps
import matplotlib.pyplot as plt
import random


# 联通数据处理
def unicom_data():
    grid_map = {}
    with open('data/250_is_500.csv', 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            d = line.strip().split(',')
            grid_map[d[1]] = d[2] # map 250 to 500
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

    flows_500 = {}
    for g, m in flows_250.items():
        if g[0] in grid_map and g[1] in grid_map:
            k = (grid_map[g[0]], grid_map[g[1]])
            if k not in flows_500:
                flows_500[k] = 0
            flows_500[k] += m

    with open('data/unicom_500.csv', 'w', newline='') as rf:
        sheet = csv.writer(rf)
        sheet.writerow(['o', 'd', 'm'])
        for g, m in flows_500.items():
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

    with open(filename[:-4]+'_c'+str(class_num)+'_t'+str(threshold)+'_'+filename[-4:], 'w', newline='') as rf:
        sheet = csv.writer(rf)
        sheet.writerow(['o', 'd', 'm', 'c'])
        for g, m in flows.items():
            x = np.where(m <= nk)[0]
            i = x.min() if x.size > 0 else len(nk) - 1
            sheet.writerow([g[0], g[1], m, 0])


def count(filename):
    data = []
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            d = line.strip().split(',')
            if d[0] == d[1] or int(d[-1]) < 50:
                line = f.readline()
                continue
            data.append(int(d[-1]))
            if int(d[-1]) == 19376:
                print(d)
            line = f.readline()

    data = np.array(data)
    print(data.shape)
    print(np.max(data), np.min(data))
    n, bins, patches = plt.hist(data, 50, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.show()



def gen_data(data_file, r, output_path):
    data = np.loadtxt(data_file, dtype=np.uint16, delimiter=',', skiprows=1)

    # 生成训练、测试、验证数据
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
    with open(output_path + 'entities.dict', 'w') as f:
        grids = set(data[:, 0]) | set(data[:, 1])
        for i, gid in enumerate(grids):
            f.write(str(i)+'\t'+ str(gid) + '\n')

    with open(output_path + 'relations.dict', 'w') as f:
        relations = set(data[:, 3])
        for i, r in enumerate(relations):
            f.write(str(i) + '\t' + str(r) + '\n')


if __name__ == '__main__':
    #unicom_data()
    #classification('data/unicom_500.csv', 3, 100)
    #count('data/unicom_500.csv')
    gen_data('data/unicom_500_c3_t100_.csv', [0.6, 0.2, 0.2], 'R-GCN/data/unicom/')