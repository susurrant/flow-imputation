import numpy as np
import matplotlib.pyplot as plt


def count(filename, gridnum):
    im = np.zeros((gridnum,gridnum), dtype=np.float64)
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

    x = im.flatten()
    d = {}
    for v in x:
        if v < 50:
            continue
        if v not in d:
            d[v] = 0
        d[v] += 1
    return d

if __name__ == '__main__':
    datafile = './data/sj_051317.csv'
    d = count(datafile, 900)
    print(d)