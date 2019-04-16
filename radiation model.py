
import numpy as np
from func import *


def read_data(path, normalization=False):
    te_f = read_flows(path + 'test.txt')
    if normalization:
        features = read_features(path + 'entities.dict', path + 'features.txt')
    else:
        features = read_features(path + 'entities.dict', path + 'features_raw.txt')

    return te_f, features


def predict(flows, features):
    real = []
    pred = []
    for f in flows:
        real.append(f[2])
        ogid = f[0]
        dgid = f[1]
        opop = features[ogid][3] + features[ogid][2]
        dpop = features[dgid][3] + features[dgid][2]
        d = dis(features[ogid][0], features[ogid][1], features[dgid][0], features[dgid][1])
        s = 0
        for gid in features:
            if gid != ogid and gid != dgid:
                if dis(features[ogid][0], features[ogid][1], features[gid][0], features[gid][1]) <= d:
                    s += features[gid][3]+features[gid][2]
        pred.append(features[ogid][3]*opop*dpop/((opop+s)*(opop+dpop+s)))

    return pred, real

if __name__ == '__main__':
    path = 'SI-GCN/data/taxi/'
    flows, features = read_data(path, False)
    pred, real = predict(flows, features)
    np.savetxt('data/pred_RM_.txt', pred, delimiter=',')
    evaluate(pred, real)