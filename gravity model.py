
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



if __name__ == '__main__':
    