
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    data = np.loadtxt(filename, dtype=np.uint16, delimiter='\t', skiprows=1)[:,[0,2,3]]
    attraction = {}
    flows = {}

    for f in data:
        fid = (min(f[0],f[1]), max(f[0], f[1]))
        if fid not in flows:
            flows[fid] = 0
        flows[fid] += f[2]

        if f[0] not in attraction:
            attraction[f[0]] = 0
        attraction[f[0]] += f[2]
        if f[1] not in attraction:
            attraction[f[1]] = 0
        attraction[f[1]] += f[2]

    return flows, attraction


def merge_attraction(entity_file, tr_a, te_a, v_a):
    gids = []
    with open(entity_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            gids.append(int(line.strip().split('\t')[-1]))

    attraction = {}
    for gid in gids:
        attraction[gid] = 0
        if gid in tr_a:
            attraction[gid] += tr_a[gid]
        if gid in te_a:
            attraction[gid] += te_a[gid]
        if gid in v_a:
            attraction[gid] += v_a[gid]
    return attraction


def grid_dis(i, j, colnum):
    x0 = int(i) % colnum
    y0 = int(i) // colnum
    x1 = int(j) % colnum
    y1 = int(j) // colnum
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def gravity_model(flows, attraction, colnum):
    Y = []
    X = []
    for k, v in flows.items():
        if v:
            #print(k)
            Y.append(np.log(attraction[k[0]]*attraction[k[1]]/v))
            X.append(np.log(grid_dis(k[0], k[1], colnum)))

    p = np.polyfit(X, Y, 1)
    beta = p[0]
    K = np.e**(-p[1])
    '''
    p1 = plt.scatter(X, Y, marker='.', color='green', s=10)
    plt.show()
    '''
    return beta, K


def evaluate(flows, attraction, beta, K, colnum):
    pred = []
    real = []
    for k, v in flows.items():
        pred.append(K * attraction[k[0]] * attraction[k[1]] / grid_dis(k[0], k[1], colnum) ** beta)
        real.append(v)

    print('real:', real[:20])
    print('pred:', pred[:20])

    print('mean absolute error:', np.mean(np.abs(np.array(pred)-np.array(real))))
    print('sum absolute error:', np.sum(np.abs(np.array(pred) - np.array(real))))



if __name__ == '__main__':
    colnum = 25
    path = 'SI-GCN/data/taxi/'
    tr_f, tr_a = read_data(path + 'train.txt')
    te_f, te_a = read_data(path + 'test.txt')
    v_f, v_a = read_data(path + 'valid.txt')
    attraction = merge_attraction(path + 'entities.dict', tr_a, te_a, v_a)

    beta, K = gravity_model(tr_f, attraction, colnum)
    print(beta, K)

    evaluate(te_f, attraction, beta, K, colnum)