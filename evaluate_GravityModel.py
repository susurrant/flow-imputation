
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    data = set()
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line:
            s = line.split('\t')
            data.add((s[0], s[2]))
            line = f.readline().strip()
    return data


def read_interaction(filename):
    flow = {}
    attract = {}
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            s = line.split(',')
            flow[(s[0], s[1])] = int(s[2])
            if s[0] not in attract:
                attract[s[0]] = 0
            attract[s[0]] += int(s[2])
            #if s[1] not in attract:
                #attract[s[1]] = 0
            #attract[s[1]] += int(s[2])
            line = f.readline().strip()

    return flow, attract


def grid_dis(i, j, colnum):
    x0 = int(i) % colnum
    y0 = int(i) // colnum
    x1 = int(j) % colnum
    y1 = int(j) // colnum
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def gravity_model(idx, flow, attract, colnum):
    Y = []
    X = []
    for k in idx:
        if k[0] in attract and k[1] in attract:
            #print(k)
            Y.append(np.log(attract[k[0]]*attract[k[1]]/flow[k]))
            X.append(np.log(grid_dis(k[0], k[1], colnum)))

    p = np.polyfit(X, Y, 1)
    beta = p[0]
    K = np.e**(-p[1])
    '''
    p1 = plt.scatter(X, Y, marker='.', color='green', s=10)
    plt.show()
    '''
    return beta, K


def evaluate(n, colnum, filtered=True):
    path = 'R-GCN/data/taxi/'
    train = read_data(path+'train.txt')
    test = read_data(path+'test.txt')
    valid = read_data(path+'valid.txt')
    flow, attract = read_interaction('data/taxi_1km_c3_t50.csv')

    beta, K = gravity_model(train, flow, attract, colnum)
    print('beta:', beta, 'K:', K)

    ranks = []
    for t in test:
        if t[1] in attract and t[0] in attract:
            dif_pred = np.abs(K*attract[t[0]]*attract[t[1]]/grid_dis(t[0], t[1], colnum)**beta - flow[t])
        else:
            continue

        # remove head
        dif_f = []
        for g in attract:
            if g == t[1]:
                continue
            if filtered:
                if (g, t[1]) not in train and (g, t[1]) not in valid and (t[0], g) not in test:
                    dif_f.append(np.abs(K*attract[t[1]]*attract[g]/grid_dis(g, t[1], colnum)**beta - flow[t]))
            else:
                dif_f.append(np.abs(K*attract[t[1]]*attract[g]/grid_dis(g, t[1], colnum)**beta - flow[t]))
        if len(dif_f) >= n:
            ranks.append(np.sum(np.array(dif_f) < dif_pred))
        else:
            print(t, len(dif_f))

        # remove tail
        dif_f = []
        for g in attract:
            if g == t[0]:
                continue
            if filtered:
                if (t[0], g) not in train and (t[0], g) not in valid and (t[0], g) not in test:
                    dif_f.append(np.abs(K*attract[t[0]]*attract[g]/grid_dis(g, t[0], colnum)**beta - flow[t]))
            else:
                dif_f.append(np.abs(K*attract[t[0]]*attract[g]/grid_dis(g, t[0], colnum)**beta - flow[t]))
        if len(dif_f) >= n:
            ranks.append(np.sum(np.array(dif_f) < dif_pred))
        else:
            print(t, len(dif_f))
    print(ranks)

    # hits at n
    hits = 0.0
    for rank in ranks:
        if rank <= n:
            hits += 1

    # MRR(Mean Reciprocal Rank)
    mrr = np.mean(1/(np.array(ranks)+1))

    return hits/len(ranks), mrr


if __name__ == '__main__':
    print(evaluate(1, colnum=25, filtered=True))
    print(evaluate(1, colnum=25, filtered=False))