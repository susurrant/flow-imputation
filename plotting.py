
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

def iter_rmse_scc():
    iv = 2
    gnn_10_rmse = np.loadtxt('data/GNN_10_RMSE.txt')[::iv]
    gnn_20_rmse = np.loadtxt('data/GNN_20_RMSE.txt')[::iv]
    gnn_30_rmse = np.loadtxt('data/GNN_30_RMSE.txt')[::iv]
    gcn_rmse = np.loadtxt('data/GCN_RMSE.txt')[::iv]

    gnn_10_scc = np.loadtxt('data/GNN_10_SMC.txt')[::iv]
    gnn_20_scc = np.loadtxt('data/GNN_20_SMC.txt')[::iv]
    gnn_30_scc = np.loadtxt('data/GNN_30_SMC.txt')[::iv]
    gcn_scc = np.loadtxt('data/GCN_SMC.txt')[::iv]

    x = np.arange(50, 10001, 50*iv)

    lw = 1
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()

    l1, = ax1.plot(x, gnn_10_rmse, color=colors[0], linewidth=lw, alpha=0.7, label='GNN_10')
    l2, = ax1.plot(x, gnn_20_rmse, color=colors[1], linewidth=lw, alpha=0.7, label='GNN_20')
    l3, = ax1.plot(x, gnn_30_rmse, color=colors[2], linewidth=lw, alpha=0.7, label='GNN_30')
    l4, = ax1.plot(x, gcn_rmse, color=colors[3], linewidth=lw, alpha=1, label='SI-GCN')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('RMSE')
    #ax1.set_xlim(0, 10000)
    #ax1.set_ylim(22, 36)

    ax2 = ax1.twinx()
    l5, = ax2.plot([30], [0.5], color='gray', linewidth=lw, linestyle='-', label='RMSE')
    l6, = ax2.plot([30], [0.5], color='gray', linewidth=lw, linestyle='--', label='SCC')
    ax2.plot(x, gnn_10_scc, color=colors[0], linestyle='--', linewidth=lw, alpha=0.7)
    ax2.plot(x, gnn_20_scc, color=colors[1], linestyle='--', linewidth=lw, alpha=0.7)
    ax2.plot(x, gnn_30_scc, color=colors[2], linestyle='--', linewidth=lw, alpha=0.7)
    ax2.plot(x, gcn_scc, color=colors[3], linestyle='--', linewidth=lw, alpha=1)

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('SCC(%)')
    #ax2.set_xlim(0, 10000)
    #ax2.set_ylim(0.2, 0.7)

    ax1.legend([l1, l2, l3, l4], labels=['GNN_10', 'GNN_20', 'GNN_30', 'SI-GCN'], bbox_to_anchor=(0.97, 0.61),
               borderaxespad=0.1, ncol=1)
    ax2.legend([l5, l6], labels=['RMSE', 'SCC'], bbox_to_anchor=(0.75, 0.55), borderaxespad=0.1, ncol=1)

    #plt.subplots_adjust(top=0.88)

    plt.show()


def check():
    r = np.loadtxt('SI-GCN/data/taxi/test.txt', dtype=np.uint32, delimiter='\t')[:, 3]
    path = 'SI-GCN/data/output/'
    files = os.listdir(path)
    for filename in files:
        p = np.loadtxt(path + filename)[:1427]
        smc = stats.spearmanr(r, p)
        scc = round(smc[0], 3)
        rmse = round(np.sqrt(np.mean(np.square(r - p))), 3)
        print(filename, scc, rmse)


def var_threshold():
    sns.set_style('whitegrid')
    real = np.loadtxt('SI-GCN/data/taxi/test.txt', dtype=np.uint32, delimiter='\t')[:, 3]
    gcn = np.loadtxt('SI-GCN/data/output/d_39000.txt')[:1427]
    gnn_10 = np.loadtxt('data/pred_gnn_10.txt')
    gnn_20 = np.loadtxt('data/pred_gnn_20.txt')
    gnn_30 = np.loadtxt('data/pred_gnn_30.txt')

    ts = [30, 40, 50, 60, 70, 80, 90, 100]
    gcn_rmse = []
    gnn_10_rmse = []
    gnn_20_rmse = []
    gnn_30_rmse = []
    for t in ts:
        idx = np.where(real >= t)
        gcn_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gcn[idx]))), 3))
        gnn_10_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gnn_10[idx]))), 3))
        gnn_20_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gnn_20[idx]))), 3))
        gnn_30_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gnn_30[idx]))), 3))

    fig, ax1 = plt.subplots()
    lw = 1
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    l1, = ax1.plot(ts, gnn_10_rmse, color=colors[0], linewidth=lw, alpha=0.7, label='GNN_10')
    l2, = ax1.plot(ts, gnn_20_rmse, color=colors[1], linewidth=lw, alpha=0.7, label='GNN_20')
    l3, = ax1.plot(ts, gnn_30_rmse, color=colors[2], linewidth=lw, alpha=0.7, label='GNN_30')
    l4, = ax1.plot(ts, gcn_rmse, color=colors[3], linewidth=lw, alpha=1, label='SI-GCN')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('RMSE')
    plt.show()


if __name__ == '__main__':
    #iter_rmse_scc()
    #check()
    var_threshold()