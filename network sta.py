
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
from sklearn import linear_model
import pysal as ps
import seaborn as sns


def imbalance_vertex(path='SI-GCN/data/taxi_0.6/'):
    intensity = np.loadtxt(path+'features_raw.txt', delimiter='\t', usecols=(2,3))
    imb_v = (intensity[:, 0]-intensity[:, 1])/(intensity[:, 0]+intensity[:, 1])
    gid = np.loadtxt(path+'entities.dict', delimiter='\t', usecols=1, dtype=np.int16)

    imb_net = np.sum(np.abs(intensity[:, 0]-intensity[:, 1]))/np.sum(intensity[:, 0]+intensity[:, 1])
    print('network imbalance:', imb_net)
    #np.savetxt('im_v.txt', list(zip(gid, imb_v)), fmt=('%d', '%.3f'), delimiter=',', header='gid,im')
    return dict(zip(gid, imb_v))


def imb_rmse(imb_v):
    real = np.loadtxt('SI-GCN/data/taxi_0.6/test.txt', delimiter='\t', dtype=np.int16)
    gcn_pred = np.loadtxt('data/output_SI-GCN/output th=30/iter_40500.txt')
    rmse_v = {}
    i = 0
    for d in real:
        if d[0] not in rmse_v:
            rmse_v[d[0]] = [0, 0]
        if d[2] not in rmse_v:
            rmse_v[d[2]] = [0, 0]
        rmse_v[d[0]][0] += (d[3] - gcn_pred[i]) ** 2
        rmse_v[d[0]][1] += 1
        rmse_v[d[2]][0] += (d[3] - gcn_pred[i]) ** 2
        rmse_v[d[2]][1] += 1

        i+=1

    rmse = []
    imb = []
    for gid in rmse_v:
        rmse.append(np.sqrt(rmse_v[gid][0]/rmse_v[gid][1]))
        imb.append(imb_v[gid])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(imb, rmse, marker='.', color='#2b83ba', s=20)
    ax.set_xlabel('b')
    ax.set_ylabel('RMSE')
    plt.show()


def gen_graph(path):
    DG = nx.DiGraph()
    data = np.loadtxt(path + 'train.txt', delimiter='\t', dtype=np.int16, usecols=(0, 2))
    DG.add_edges_from(data.tolist())
    data = np.loadtxt(path + 'valid.txt', delimiter='\t', dtype=np.int16, usecols=(0, 2))
    DG.add_edges_from(data.tolist())
    data = np.loadtxt(path + 'test.txt', delimiter='\t', dtype=np.int16, usecols=(0, 2))
    DG.add_edges_from(data.tolist())

    return DG


def network_sta(path):
    DG = gen_graph(path)
    print(DG.number_of_nodes(), DG.number_of_edges())
    G = DG.to_undirected()

    n = 0
    s = 0
    edge_num = 0
    node_num = 0

    for g in nx.connected_component_subgraphs(G):
        node_num = g.number_of_nodes()
        aspl = nx.average_shortest_path_length(g)
        print('subgraph:', node_num, g.number_of_edges(), aspl)
        if node_num < 10:
            nodes = list(g.nodes)
            nn = len(nodes)
            node_num += nn
            for i in range(nn):
                for j in range(nn):
                    if DG.has_edge(nodes[i], nodes[j]):
                        edge_num += 1
        n += node_num
        s += aspl*node_num
    print('edge ratio:', edge_num, 1-edge_num/DG.number_of_edges())
    print('node ratio:', node_num, 1-node_num/DG.number_of_nodes())
    aspl = s/n
    acc = nx.average_clustering(G)
    print('\nnet aspl:', aspl)
    print('acc:', acc)

    SG = nx.fast_gnp_random_graph(n, G.number_of_edges()/(n*(n-1)/2))
    sacc = nx.average_clustering(SG)
    saspl = nx.average_shortest_path_length(SG)
    print('\ns net aspl:', saspl)
    print('s acc:', sacc)

    print('delta:', acc/sacc/(aspl/saspl))


def degree_distribution(path):
    DG = gen_graph(path)
    G = DG.to_undirected()
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='#2b83ba')

    plt.ylabel("Count")
    plt.xlabel("Degree")
    #ax.set_xticks([d + 0.4 for d in deg])
    #ax.set_xticklabels(deg)

    plt.axes([0.4, 0.4, 0.5, 0.5])
    log_deg = np.log10(deg).reshape(-1, 1)
    log_cnt = np.log10(cnt).reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(log_deg, log_cnt)
    print('Coefficients: \n', regr.coef_, )
    print("Intercept:\n", regr.intercept_)
    print(regr.score(log_deg, log_cnt))

    plt.scatter(log_deg, log_cnt, color='#2b83ba', s=10)
    plt.plot(log_deg, regr.predict(log_deg), color='#abdda4', linewidth=1)
    plt.text(0.7, 1.5, r'y=-0.892x+0.722 (R2=0.722)', fontdict={'size': '10', 'color': 'black'})

    plt.show()


def intensity_distance_distribution(path, col_num=30):
    dis_list = []
    intensity_list = []
    flows = np.loadtxt(path + 'train.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
        intensity_list.append(int(d[3]))
    flows = np.loadtxt(path + 'test.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
        intensity_list.append(int(d[3]))
    flows = np.loadtxt(path + 'valid.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
        intensity_list.append(int(d[3]))
    print(min(dis_list), max(dis_list))
    dis = [i*0.5 for i in range(14)]
    cnt_dis = [0]*len(dis)
    for d in dis_list:
        cnt_dis[int(d)-1] += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.hist(intensity_list, bins=100, color='#2b83ba')
    #ax1.bar(itn, cnt_itn)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Intensity")
    ax1.set_xticks([50*i for i in range(1, 10)])
    ax1.set_yticks([200 * i for i in range(1, 8)])
    ax1.set_ylim(0,1450)

    print(dis)
    print(cnt_dis)
    ax2 = fig.add_subplot(212)
    ax2.bar(dis, cnt_dis, width=0.5, color='#2b83ba')
    #ax2.hist(dis_list)
    ax2.set_xticks(np.array(dis)-0.25)
    ax2.set_xticklabels([i for i in range(1, 15)])
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Distance/km")
    ax2.set_ylim(0, 2300)
    plt.show()


def cumulative_intensity(path):
    intensity_list = []
    flows = np.loadtxt(path + 'train.txt', delimiter='\t')
    for d in flows:
        intensity_list.append(int(d[3]))
    flows = np.loadtxt(path + 'test.txt', delimiter='\t')
    for d in flows:
        intensity_list.append(int(d[3]))
    flows = np.loadtxt(path + 'valid.txt', delimiter='\t')
    for d in flows:
        intensity_list.append(int(d[3]))

    print(max(intensity_list))
    itn = [i for i in range(30, 474)]
    cnt_itn = [0] * len(itn)
    for i in intensity_list:
        cnt_itn[i - 30] += 1

    t_itn = sum(cnt_itn)
    cum_itn = [sum(cnt_itn[:i]) / t_itn for i in range(1, len(itn) + 1)]
    print(cnt_itn[:10])
    print(cum_itn[:10])

    fig = plt.figure(figsize=(4.5,2))
    ax = fig.add_subplot(111)
    ax.plot(itn, cum_itn, color='#2b83ba', linewidth=1)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('CFreq')
    ax.set_xlim(20, 475)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    plt.show()


def cumulative_distance(path, col_num=30):
    dis_list = []
    flows = np.loadtxt(path + 'train.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
    flows = np.loadtxt(path + 'test.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
    flows = np.loadtxt(path + 'valid.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))

    print(min(dis_list), max(dis_list))
    dis = [i * 0.5 for i in range(14)]
    cnt_dis = [0] * len(dis)
    for d in dis_list:
        cnt_dis[int(d) - 1] += 1

    t_itn = sum(cnt_dis)
    cum_itn = [sum(cnt_dis[:i]) / t_itn for i in range(1, len(dis) + 1)]

    fig = plt.figure(figsize=(4.5,2))
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(1, 15)], cum_itn, color='#2b83ba', linewidth=1)
    ax.set_xlabel('Distance/km')
    ax.set_ylabel('CFreq')
    #ax.set_xlim(20, 475)
    ax.set_ylim(0.25, 1.05)
    ax.set_yticks([0.2, 0.6, 1.0])
    plt.show()


def grid_dis(i, j, colnum):
    x0 = int(i) % colnum
    y0 = int(i) // colnum
    x1 = int(j) % colnum
    y1 = int(j) // colnum
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def fisher_jenks(d, cNum):
    X = np.array(d).reshape((-1, 1))
    fj = ps.esda.mapclassify.Fisher_Jenks(X, cNum)
    meanV = []
    for i in range(cNum):
        meanV.append(np.mean(X[np.where(i == fj.yb)]))
    return fj.bins, meanV


def dis_intensity(path, col_num=30, class_num=5):
    dis_list = []
    intensity_list = []
    flows = np.loadtxt(path + 'train.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
        intensity_list.append(int(d[3]))
    flows = np.loadtxt(path + 'test.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
        intensity_list.append(int(d[3]))
    flows = np.loadtxt(path + 'valid.txt', delimiter='\t')
    for d in flows:
        dis_list.append(grid_dis(d[0], d[2], col_num))
        intensity_list.append(int(d[3]))

    nk, nl = fisher_jenks(dis_list, class_num)
    print(nk)
    cnt = [0]*class_num
    intensity = [0]*class_num
    for i, dis in enumerate(dis_list):
        x = np.where(dis <= nk)[0]
        idx = x.min() if x.size > 0 else len(nk) - 1
        cnt[idx] += 1
        intensity[idx] += intensity_list[i]

    for i in range(class_num):
        intensity[i] /= cnt[i]
    print(intensity)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    ax.bar([1,2,3,4,5], intensity, facecolor='#2b83ba', width=0.5)
    ax.set_ylim(0, 80)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.xaxis.set_ticklabels(['<1.4', '1.4-2.8', '2.8-4.5', '4.5-7.3', '≥7.3'])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel('Distance/km')
    ax.set_ylabel('Flow intensity')
    plt.show()


if __name__ == '__main__':
    path = 'SI-GCN/data/taxi_0.6/'

    #imb_v = imbalance_vertex()
    #imb_rmse(imb_v)

    network_sta(path)

    #degree_distribution(path)

    #intensity_distance_distribution(path)

    #cumulative_intensity(path)
    #cumulative_distance(path)

    #dis_intensity(path)