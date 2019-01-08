# -*- coding: utf-8 -*-：

import networkx as nx
import matplotlib.pyplot as plt

def construct_net_from_file(filename, minSpeed = 2, maxSpeed = 150):
    si = {}
    nodes = set()
    with open(filename, 'r') as f:
        f.readline()
        while True:
            line1 = f.readline().strip()
            if line1:
                sl1 = line1.split(',')
                sl2 = f.readline().strip().split(',')
                if sl1[1] == '1' and minSpeed < float(sl1[-2]) < maxSpeed:
                    ogid = int(sl1[-1])
                    dgid = int(sl2[-1])
                    nodes.add(ogid)
                    nodes.add(dgid)
                    if (ogid, dgid) not in si:
                        si[(ogid, dgid)] = 0
                    si[(ogid, dgid)] += 1
            else:
                break

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for k, v in si.items():
        G.add_weighted_edges_from([(k[0], k[1], v)])

    return G

if __name__ == '__main__':
    G = construct_net_from_file('./data/sj_051316_1km.csv')
    nx.draw(G)
    plt.show()
    '''
    degree = nx.degree_histogram(G)
    x = []
    y = []
    for i in range(len(degree)):
        if degree[i]:
            x.append(i)
            y.append(degree[i])

    #print(degree)
    #x = range(len(degree))  # 生成x轴序列，从1到最大度
    #y = [z / float(sum(degree)) for z in degree]
    # 将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
    #plt.loglog(x, y, color="blue", linewidth=2)  # 在双对数坐标轴上绘制度分布曲线
    plt.scatter(x, y)
    plt.show()
    '''