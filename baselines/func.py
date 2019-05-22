
import numpy as np
from scipy import stats
#import matplotlib.pyplot as plt


def read_flows(filename):
    flows = np.loadtxt(filename, dtype=np.uint32, delimiter='\t')[:,[0,2,3]]
    return flows


def read_features(entity_dict, feature_file):
    grid_list = list(np.loadtxt(entity_dict, dtype=np.uint32, delimiter='\t')[:, 1])
    features = {}
    with open(feature_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sl = line.strip().split('\t')
            features[grid_list[i]] = list(map(float, sl))  #[x, y, attract, pull]
    return features


def grid_dis(i, j, colnum):
    x0 = int(i) % colnum
    y0 = int(i) // colnum
    x1 = int(j) % colnum
    y1 = int(j) // colnum
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def dis(x0, y0, x1, y1):
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def evaluate(p, r, mode='positive'):
    print('\nnum of test flows:', len(r))
    print('real_min:', min(r), ', real_max:', max(r))
    print('pred_min:', int(min(p)), ', pred_max:', int(max(p)))
    print('real:', r[0:20])
    print('pred:', list(map(int, p[0:20])))

    p = np.array(p)
    r = np.array(r)

    if mode == 'positive':
        #print('MAE:', round(np.mean(np.abs(r - p)),3))

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

        #print('MSE:', round(np.mean(np.square(r - p)), 3))
        print('RMSE:', round(np.sqrt(np.mean(np.square(r - p))), 3))

        stack = np.column_stack((p, r))
        print('CPC:', round(2 * np.sum(np.min(stack, axis=1)) / np.sum(stack), 3))

        #print('SSI:', round(ssi * 2 / (c2 ^ 2), 3))

        smc = stats.spearmanr(r, p)
        print('SMC: correlation =', round(smc[0], 3), ', p-value =', round(smc[1], 3))

        llr = stats.linregress(r, p)
        #print('LLR: R =', round(llr[2], 3), ', p-value =', round(llr[3], 3))
    elif mode == 'negative':
        print('Proportion of zeros:', round(np.sum(p < 0.5) / p.shape[0], 3))
        print('MAE:', round(np.mean(np.abs(r - p)), 3))
        print('RMSE:', round(np.sqrt(np.mean(np.square(r - p))), 3))


