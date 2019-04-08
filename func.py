
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def read_flows(filename):
    flows = np.loadtxt(filename, dtype=np.uint16, delimiter='\t')[:,[0,2,3]]
    return flows


def read_features(entity_dict, feature_file):
    grid_list = list(np.loadtxt(entity_dict, dtype=np.uint16, delimiter='\t')[:, 1])
    features = {}
    with open(feature_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sl = line.strip().split('\t')
            features[grid_list[i]] = list(map(float, sl))  #[attract, pull]
    return features


def grid_dis(i, j, colnum):
    x0 = int(i) % colnum
    y0 = int(i) // colnum
    x1 = int(j) % colnum
    y1 = int(j) // colnum
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def dis(x0, y0, x1, y1):
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def evaluate(p, r):
    p = np.array(p)
    r = np.array(r)

    print(np.sum(p != 0) / p.shape[0])

    print('\nMAE\t', round(np.mean(np.abs(r - p)),3))

    c1 = 0
    mape = 0
    c2 = 0
    ssi = 0
    for i in range(p.shape[0]):
        if r[i]:
            mape += np.abs((r[i] - p[i]) / r[i])
            c1 += 1
        if r[i] + p[i]:
            ssi += min(r[i], p[i]) / (r[i] + p[i])
            c2 += 1
    print('MAPE:', round(mape / c1, 3))

    print('MSE:', round(np.mean(np.square(r - p)), 3))
    print('RMSE:', round(np.sqrt(np.mean(np.square(r - p))), 3))

    stack = np.column_stack((p, r))
    print('CPC:', round(2 * np.sum(np.min(stack, axis=1)) / np.sum(stack), 3))

    print('SSI:', round(ssi * 2 / (c2 ^ 2), 3))

    smc = stats.spearmanr(r, p)
    print('SMC: correlation =', round(smc[0], 3), ', p-value =', round(smc[1], 3))

    llr = stats.linregress(r, p)
    print('LLR: R =', round(llr[2], 3), ', p-value =', round(llr[3], 3))

    #p1 = plt.scatter(p, r, marker='.', color='green', s=10)
    #plt.show()

