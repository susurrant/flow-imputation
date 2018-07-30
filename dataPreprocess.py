
import numpy as np
import random
import os

def read_data(filename, grid_num, threshold):
    im = np.zeros((grid_num, grid_num), dtype=np.uint16)
    with open(filename, 'r') as f:
        f.readline()
        line1 = f.readline().strip()
        while line1:
            sl1 = line1.split(',')
            line2 = f.readline().strip()
            sl2 = line2.split(',')
            if int(sl1[1]) == 1 and int(sl2[1]) == 1:
                im[int(sl1[-1]), int(sl2[-1])] += 1
            line1 = f.readline().strip()

    x, y = np.where(im >= threshold)
    return list(zip(list(map(str, x)), list(map(str, y))))


def output_data(si, p, path):
    n = len(si)
    train_set = random.sample(si, int(p[0]*n))
    with open(path+'train.txt', 'w') as f:
        for d in train_set:
            f.write(d[0]+'\t'+ '_to'+ '\t' + d[1] + '\n')
            si.remove(d)

    test_set = random.sample(si, int(p[1]*n))
    with open(path+'test.txt', 'w') as f:
        for d in test_set:
            f.write(d[0]+'\t'+ '_to'+ '\t' + d[1] + '\n')
            si.remove(d)

    with open(path+'valid.txt', 'w') as f:
        for d in si:
            f.write(d[0]+'\t'+ '_to'+ '\t' + d[1] + '\n')


def output_dict(grid_num, path):
    with open(path+'entities.dict', 'w') as f:
        for i in range(grid_num):
            f.write(str(i)+'\t'+ str(i) + '\n')

    with open(path+'relations.dict', 'w') as f:
        f.write('0\t'+ '_to\n')


if __name__ == '__main__':
    threshold = 60
    path = './R-GCN/data/SI_' + str(threshold) + '/'
    data_file = './data/sj_051317.csv'
    if not os.path.exists(path):
        os.mkdir(path)

    grid_num = 900
    p = (0.8, 0.1, 0.1)
    si = read_data(data_file, grid_num, threshold)
    output_data(si, p, path)
    output_dict(grid_num, path)