
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def iter_rmse_scc():
    #sns.set_style('whitegrid')
    iv = 2
    gnn_rmse_10 = np.loadtxt('GNN_RMSE_10.txt')[::iv]
    gnn_rmse_20 = np.loadtxt('GNN_RMSE_20.txt')[::iv]
    gnn_rmse_30 = np.loadtxt('GNN_RMSE_30.txt')[::iv]
    gcn_rmse = np.loadtxt('GCN_RMSE.txt')[::iv]

    gnn_scc_10 = np.loadtxt('GNN_SMC_10.txt')[::iv]
    gnn_scc_20 = np.loadtxt('GNN_SMC_20.txt')[::iv]
    gnn_scc_30 = np.loadtxt('GNN_SMC_30.txt')[::iv]
    gcn_scc = np.loadtxt('GCN_SMC.txt')[::iv]

    x = np.arange(50, 10001, 50*iv)

    lw = 1
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()

    l1, = ax1.plot(x, gnn_rmse_10, color=colors[0], linewidth=lw, alpha=0.7, label='GNN_10')
    l2, = ax1.plot(x, gnn_rmse_20, color=colors[1], linewidth=lw, alpha=0.7, label='GNN_20')
    l3, = ax1.plot(x, gnn_rmse_30, color=colors[2], linewidth=lw, alpha=0.7, label='GNN_30')
    l4, = ax1.plot(x, gcn_rmse, color=colors[3], linewidth=lw, alpha=1, label='SI-GCN')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('RMSE')
    #ax1.set_xlim(0, 10000)
    #ax1.set_ylim(22, 36)

    ax2 = ax1.twinx()
    l5, = ax2.plot([30], [0.5], color='gray', linewidth=lw, linestyle='-', label='RMSE')
    l6, = ax2.plot([30], [0.5], color='gray', linewidth=lw, linestyle='--', label='SCC')
    ax2.plot(x, gnn_scc_10, color=colors[0], linestyle='--', linewidth=lw, alpha=0.7)
    ax2.plot(x, gnn_scc_20, color=colors[1], linestyle='--', linewidth=lw, alpha=0.7)
    ax2.plot(x, gnn_scc_30, color=colors[2], linestyle='--', linewidth=lw, alpha=0.7)
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

if __name__ == '__main__':
    iter_rmse_scc()