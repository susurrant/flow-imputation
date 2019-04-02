
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
            features[grid_list[i]] = list(map(int, sl[2:]))  #[attract, pull]
    return features


def grid_dis(i, j, colnum):
    x0 = int(i) % colnum
    y0 = int(i) // colnum
    x1 = int(j) % colnum
    y1 = int(j) // colnum
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def gravity_model(flows, features, colnum):
    Y = []
    X = []
    feature_size = len(list(features.values())[0])
    if feature_size == 1:
        for k in flows:
            if k[2]:
                Y.append(np.log(features[k[0]] * features[k[1]] / k[2]))
                X.append(np.log(grid_dis(k[0], k[1], colnum)))
    elif feature_size == 2:
        for k in flows:
            if k[2]:
                Y.append(np.log(features[k[0]][1] * features[k[1]][0] / k[2]))
                X.append(np.log(grid_dis(k[0], k[1], colnum)))

    p = np.polyfit(X, Y, 1)
    beta = p[0]
    K = np.e**(-p[1])

    #p1 = plt.scatter(X, Y, marker='.', color='green', s=10)
    #plt.show()

    return beta, K


def predict(flows, features, beta, K, colnum):
    p = []
    r = []
    feature_size = len(list(features.values())[0])
    if feature_size == 1:
        for f in flows:
            p.append(K * features[f[0]] * features[f[1]] / grid_dis(f[0], f[1], colnum) ** beta)
            r.append(f[2])
    elif feature_size == 2:
        for f in flows:
            p.append(K * features[f[0]][1] * features[f[1]][0] / grid_dis(f[0], f[1], colnum) ** beta)
            r.append(f[2])

    print('\nnum of test flows:', len(r))
    print('real_min:', min(r), ', real_max:', max(r))
    print('pred_min:', int(min(p)), ', pred_max:', int(max(p)))
    print('real:', r[-20:])
    print('pred:', list(map(int, p[-20:])))

    return p, r


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
    #print('MAPE:', round(mape / c1, 3))

    print('MSE:', round(np.mean(np.square(r - p)), 3))
    print('RMSE:', round(np.sqrt(np.mean(np.square(r - p))), 3))

    stack = np.column_stack((p, r))
    print('CPC:', round(2 * np.sum(np.min(stack, axis=1)) / np.sum(stack), 3))

    print('SSI:', round(ssi * 2 / (c2 ^ 2), 3))

    #smc = stats.spearmanr(r, p)
    #print('SMC: correlation =', round(smc[0], 3), ', p-value =', round(smc[1], 3))

    #llr = stats.linregress(r, p)
    #print('LLR: R =', round(llr[2], 3), ', p-value =', round(llr[3], 3))

    #p1 = plt.scatter(p, r, marker='.', color='green', s=10)
    #plt.show()


if __name__ == '__main__':

    col_num = 30
    path = 'SI-GCN/data/taxi/'
    tr_f = read_flows(path + 'train.txt')
    te_f = read_flows(path + 'test_n.txt')
    #v_f = read_flows(path + 'valid.txt')
    features = read_features(path + 'entities.dict', path + 'features_raw.txt')

    beta, K = gravity_model(tr_f, features, col_num)
    print('beta =', beta, ', K =', K)
    pred, real = predict(te_f, features, beta, K, col_num)
    evaluate(pred, real)
