# -*- coding: utf-8 -*-：

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from baselines.func import grid_dis
import pysal as ps

def iter_rmse_scc():
    iv = 2
    gnn_10_rmse = np.loadtxt('data/output_baselines/GNN_10_RMSE.txt')[::iv]
    gnn_20_rmse = np.loadtxt('data/output_baselines/GNN_20_RMSE.txt')[::iv]
    gnn_30_rmse = np.loadtxt('data/output_baselines/GNN_30_RMSE.txt')[::iv]
    gcn_rmse = np.loadtxt('data/output_SI-GCN/GCN_RMSE_th30.txt')[::iv]

    gnn_10_scc = np.loadtxt('data/output_baselines/GNN_10_SMC.txt')[::iv]
    gnn_20_scc = np.loadtxt('data/output_baselines/GNN_20_SMC.txt')[::iv]
    gnn_30_scc = np.loadtxt('data/output_baselines/GNN_30_SMC.txt')[::iv]
    gcn_scc = np.loadtxt('data/output_SI-GCN/GCN_SMC_th30.txt')[::iv]

    x = np.arange(50, 10001, 50*iv)

    lw = 1
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()

    l1, = ax1.plot(x, gnn_10_rmse, color=colors[0], linewidth=lw, alpha=0.7, label='GNN_10')
    l2, = ax1.plot(x, gnn_20_rmse, color=colors[1], linewidth=lw, alpha=0.7, label='GNN_20')
    l3, = ax1.plot(x, gnn_30_rmse, color=colors[2], linewidth=lw, alpha=0.7, label='GNN_30')
    l4, = ax1.plot(x, gcn_rmse, color=colors[3], linewidth=lw, alpha=1, label='SI-GCN')
    ax1.set_xlabel('Iteration')
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
    ax2.set_ylabel('SCC')
    #ax2.set_xlim(0, 10000)
    #ax2.set_ylim(0.2, 0.7)

    ax1.legend([l1, l2, l3, l4], labels=['GNN_10', 'GNN_20', 'GNN_30', 'SI-GCN'], bbox_to_anchor=(0.97, 0.61),
               borderaxespad=0.1, ncol=1)
    ax2.legend([l5, l6], labels=['RMSE', 'SCC'], bbox_to_anchor=(0.75, 0.55), borderaxespad=0.1, ncol=1)

    #plt.subplots_adjust(top=0.88)

    plt.show()


# determine which output produces the best prediction accuracy
def check():
    r = np.loadtxt('SI-GCN/data/taxi/test.txt', dtype=np.uint32, delimiter='\t')[:, 3]
    path = 'SI-GCN/data/output/'
    files = os.listdir(path)
    for filename in files:
        p = np.loadtxt(path + filename, delimiter=',')
        smc = stats.spearmanr(r, p)
        scc = round(smc[0], 3)
        rmse = round(np.sqrt(np.mean(np.square(r - p))), 3)
        print(filename, scc, rmse)


def var_intensity():
    sns.set_style('whitegrid')
    real = np.loadtxt('SI-GCN/data/taxi/test.txt', dtype=np.uint32, delimiter='\t')[:, 3]
    gcn = np.loadtxt('SI-GCN/data/output/iter_39000.txt', delimiter=',')
    #gnn_10 = np.loadtxt('data/pred_gnn_10.txt')
    #gnn_20 = np.loadtxt('data/pred_gnn_20.txt')
    gm_p = np.loadtxt('data/pred_GM_P.txt')
    rm = np.loadtxt('data/pred_RM.txt')
    gnn_30 = np.loadtxt('data/pred_gnn_30.txt')

    ts = [30, 40, 50, 60, 70, 80, 90, 100]
    gcn_rmse = []
    gnn_10_rmse = []
    gnn_20_rmse = []
    gnn_30_rmse = []
    gm_p_rmse = []
    rm_rmse = []
    for t in ts:
        idx = np.where(real >= t)
        gcn_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gcn[idx]))), 3))
        #gnn_10_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gnn_10[idx]))), 3))
        #gnn_20_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gnn_20[idx]))), 3))
        gnn_30_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gnn_30[idx]))), 3))
        gm_p_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - gm_p[idx]))), 3))
        rm_rmse.append(round(np.sqrt(np.mean(np.square(real[idx] - rm[idx]))), 3))

    fig, ax1 = plt.subplots()
    lw = 1
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    #l1, = ax1.plot(ts, gnn_10_rmse, color=colors[0], linewidth=lw, alpha=0.7, label='GNN_10')
    #l2, = ax1.plot(ts, gnn_20_rmse, color=colors[1], linewidth=lw, alpha=0.7, label='GNN_20')
    l1, = ax1.plot(ts, rm_rmse, color=colors[0], linewidth=lw, alpha=1, label='RM')
    l2, = ax1.plot(ts, gm_p_rmse, color=colors[1], linewidth=lw, alpha=1, label='GM_P')
    l3, = ax1.plot(ts, gnn_30_rmse, color=colors[2], linewidth=lw, alpha=0.7, label='GNN_30')
    l4, = ax1.plot(ts, gcn_rmse, color=colors[3], linewidth=lw, alpha=1, label='SI-GCN')

    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('RMSE')
    ax1.set_ylim(20, 110)
    ax1.set_xlim(30, 100)
    ax1.legend([l1, l2, l3, l4], labels=['RM', 'GM_P', 'GNN_30', 'SI-GCN'], loc='upper left', #bbox_to_anchor=(0.97, 0.61),
               borderaxespad=0.1, ncol=1)
    plt.show()


def var_distance():
    # --------------------data process----------------------
    real = np.loadtxt('SI-GCN/data/taxi_th30/test.txt', dtype=np.uint32, delimiter='\t')
    dis_list = []
    for d in real:
        dis_list.append(grid_dis(d[0], d[2], 30))

    nk, nl = fisher_jenks(dis_list, 3)
    print(nk, nl)
    dis_idx = []
    for i, dis in enumerate(dis_list):
        x = np.where(dis <= nk)[0]
        dis_idx.append(x.min() if x.size > 0 else len(nk) - 1)
    dis_idx = np.array(dis_idx)

    gcn = np.loadtxt('data/output_SI-GCN/output_190616/iter_39000.txt')
    gm_p = np.loadtxt('data/output_baselines/pred_GM_P.txt')
    rm = np.loadtxt('data/output_baselines/pred_RM.txt')
    gnn_30 = np.loadtxt('data/output_baselines/pred_GNN_30.txt')

    # short, medium, long;
    gcn_rmse = []
    gnn_30_rmse = []
    gm_p_rmse = []
    rm_rmse = []
    for i in range(3):
        idx = np.where(dis_idx == i)
        print(i, len(idx[0]), dis_list[idx[0][0]], np.mean(real[:, 3][idx]))
        gcn_rmse.append(round(np.sqrt(np.mean(np.square(real[:, 3][idx] - gcn[idx]))), 3))
        gnn_30_rmse.append(round(np.sqrt(np.mean(np.square(real[:, 3][idx] - gnn_30[idx]))), 3))
        gm_p_rmse.append(round(np.sqrt(np.mean(np.square(real[:, 3][idx] - gm_p[idx]))), 3))
        rm_rmse.append(round(np.sqrt(np.mean(np.square(real[:, 3][idx] - rm[idx]))), 3))
    rmse = [gcn_rmse, gnn_30_rmse, gm_p_rmse, rm_rmse]
    print(rmse)

    # ------------------------draw----------------------------
    colors = ['skyblue', 'limegreen', 'hotpink', 'orangered']
    #colors = ['#DDEAF6', '#8FACBD', '#F0DBD1', '#BDB6B4']
    labels = ['SI-GCN', 'GNN_30', 'GM_P', 'RM']
    sns.set(style="whitegrid")

    x = np.array([0.5, 1.5, 2.5])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlabel('Distance')
    ax.set_ylabel('RMSE')

    bw = 0.11
    ll = []
    for n, i in enumerate([-3, -1, 1, 3]):
        ll.append(ax.bar(x + i * bw / 2, rmse[n], facecolor=colors[n], width=bw, label=labels[n], alpha=0.7))

    ax.set_ylim(0, 65)

    ax.set_xlim(0, 3)
    xs = [0.5, 1.5, 2.5]
    ax.set_xticks(xs)
    xlabels = ['Short', 'Medium', 'Long']
    ax.xaxis.set_ticklabels(xlabels)

    leg = plt.legend(handles=ll)
    for l in leg.get_texts():
        l.set_fontsize(10)
        l.set_fontname('Arial')
    plt.show()


def limited_attributes_old():
    # GCN_limited, GNN_30, GCN

    RMSE = [[25.326,25.361,25.523,25.644,25.632,25.488,25.543,25.43,25.953,25.332],
            [25.154,25.107,25.116,25.105,25.071,25.055,24.992,24.936,25.137,25.147],
            [20.516,20.62,20.854,20.191,21.055,20.826,20.546,21.049,20.893]]

    MAPE = [np.array([0.306,0.294,0.305,0.319,0.329,0.321,0.313,0.311,0.303,0.301])*100,
            np.array([0.279,0.277,0.277,0.276,0.276,0.277,0.276,0.276,0.277,0.277])*100,
            np.array([0.234,0.231,0.225,0.226,0.226,0.229,0.236,0.225,0.237])*100]

    SCC = [[0.621,0.614,0.612,0.613,0.616,0.621,0.613,0.616,0.612,0.612],
           [0.647,0.654,0.649,0.649,0.648,0.653,0.653,0.659,0.646,0.648],
           [0.709,0.708,0.716,0.715,0.713,0.701,0.703,0.702,0.712]]

    CPC = [[0.855,0.856,0.854,0.853,0.852,0.855,0.854,0.855,0.853,0.855],
           [0.86,0.861,0.86,0.861,0.861,0.861,0.861,0.861,0.86,0.86],
           [0.884,0.883,0.884,0.886,0.884,0.882,0.883,0.882,0.882]]


    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    colors = ['grey']*4

    fig = plt.figure()
    ax1 = fig.add_subplot(141)
    bp = ax1.boxplot(RMSE, sym='', widths=0.5)
    set_box_color(bp, colors[0])
    ax1.set_ylabel('RMSE', fontname = 'Arial')

    ax2 = fig.add_subplot(142)
    bp = ax2.boxplot(MAPE, sym='', widths=0.5)
    set_box_color(bp, colors[1])
    ax2.set_ylabel('MAPE(%)', fontname='Arial')

    ax3 = fig.add_subplot(143)
    bp = ax3.boxplot(SCC, sym='', widths=0.5)
    set_box_color(bp, colors[2])
    ax3.set_ylabel('SCC', fontname='Arial')

    ax4 = fig.add_subplot(144)
    bp = ax4.boxplot(CPC, sym='', widths=0.5)
    set_box_color(bp, colors[3])
    ax4.set_ylabel('CPC', fontname='Arial')
    plt.figtext(0.15, 0.05, '1:SI-GCNs with limited attributes, 2:GNN_30 with entire attributes',
                fontdict={'family':'Arial', 'size':12})
    plt.figtext(0.35, 0.005, '3:SI-GCNs with entire attributes', fontdict={'family':'Arial', 'size':12})
    plt.show()


def limited_attributes():
    # GCN_limited, GNN_30, GCN

    RMSE = [[25.326,25.361,25.523,25.644,25.632,25.488,25.543,25.43,25.953,25.332],
            [25.154,25.107,25.116,25.105,25.071,25.055,24.992,24.936,25.137,25.147],
            [20.516,20.62,20.854,20.191,21.055,20.826,20.546,21.049,20.893]]

    MAPE = [np.array([0.306,0.294,0.305,0.319,0.329,0.321,0.313,0.311,0.303,0.301])*100,
            np.array([0.279,0.277,0.277,0.276,0.276,0.277,0.276,0.276,0.277,0.277])*100,
            np.array([0.234,0.231,0.225,0.226,0.226,0.229,0.236,0.225,0.237])*100]

    SCC = [[0.621,0.614,0.612,0.613,0.616,0.621,0.613,0.616,0.612,0.612],
           [0.647,0.654,0.649,0.649,0.648,0.653,0.653,0.659,0.646,0.648],
           [0.709,0.708,0.716,0.715,0.713,0.701,0.703,0.702,0.712]]

    CPC = [[0.855,0.856,0.854,0.853,0.852,0.855,0.854,0.855,0.853,0.855],
           [0.86,0.861,0.86,0.861,0.861,0.861,0.861,0.861,0.86,0.86],
           [0.884,0.883,0.884,0.886,0.884,0.882,0.883,0.882,0.882]]


    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    #colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    colors = ['grey']*4

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    bp = ax1.boxplot(RMSE, sym='', widths=0.5)
    set_box_color(bp, colors[0])
    ax1.set_ylabel('RMSE', fontname = 'Arial')

    ax2 = fig.add_subplot(122)
    bp = ax2.boxplot(MAPE, sym='', widths=0.5)
    set_box_color(bp, colors[1])
    ax2.set_ylabel('MAPE(%)', fontname='Arial')

    plt.figtext(0.15, 0.05, '1:SI-GCNs using limited attributes, 2:GNN_30 using entire attributes',
                fontdict={'family':'Arial', 'size':12})
    plt.figtext(0.35, 0.005, '3:SI-GCNs using entire attributes', fontdict={'family':'Arial', 'size':12})
    plt.show()


def var_threshold():
    GM_P_rmse = [18.491, 24.273, 27.592, 32.258, 35.698]
    GM_P_scc = [0.743, 0.686, 0.643, 0.614, 0.557]
    GM_P_mape = [34, 31, 27.7, 26.6, 24.4]
    RM_rmse = [40.475, 52.843, 55.547, 69.737, 79.818]
    RM_scc = [0.735, 0.675, 0.624, 0.59, 0.581]
    RM_mape = [90.4, 86.1, 83.5, 81.4, 80.7]
    GNN_30_rmse = [15.091, 20.973, 25.082, 28.751, 34.028]
    GNN_30_scc = [0.755, 0.696, 0.651, 0.615, 0.559]
    GNN_30_mape = [33.1, 30.2, 27.7, 26.7, 24.3]
    SI_GCN_rmse = [12.442, 16.655, 19, 24.219, 26.706]  # replace 19
    SI_GCN_scc = [0.77, 0.735,0.69, 0.657, 0.666]       # replace 0.69
    SI_GCN_mape = [30.2, 26.6, 24.8, 23.3, 20.1]        # replace 24.8



    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()
    bw = 0.11
    bar_alpha = 0.6
    l1 = ax1.bar(x - 3 * bw / 2, SI_GCN_rmse, facecolor=colors[3], width=bw, label='SI-GCN', alpha=bar_alpha)
    l2 = ax1.bar(x - 1 * bw / 2, GNN_30_rmse, facecolor=colors[2], width=bw, label='GNN_30', alpha=bar_alpha)
    l3 = ax1.bar(x + 1 * bw / 2, GM_P_rmse, facecolor=colors[1], width=bw, label='GM_P', alpha=bar_alpha)
    l4 = ax1.bar(x + 3 * bw / 2, RM_rmse, facecolor=colors[0], width=bw, label='RM', alpha=bar_alpha)

    ax1.set_xlabel('Taxi trip threshold', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax1.xaxis.set_ticklabels(['10', '20', '30', '40', '50'], fontsize=12)
    ax1.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    ax1.yaxis.set_ticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80'], fontsize=12)

    # ax1.set_xlim(0, 10000)
    # ax1.set_ylim(22, 36)

    lw = 1
    line_alpha = 0.7
    ax2 = ax1.twinx()
    l5, = ax2.plot(x, SI_GCN_scc, color=colors[3], linestyle='--', marker='s', linewidth=lw, alpha=line_alpha, label='SI-GCN')
    l6, = ax2.plot(x, GNN_30_scc, color=colors[2], linestyle='--', marker='^', linewidth=lw, alpha=line_alpha, label='GNN_30')
    l7, = ax2.plot(x, GM_P_scc, color=colors[0], linestyle='--', marker='.', linewidth=lw, alpha=line_alpha, label='GM_P')
    l8, = ax2.plot(x, RM_scc, color=colors[1], linestyle='--', marker='H', linewidth=lw, alpha=line_alpha, label='RM')


    #ax2.set_xlabel('Taxi trip threshold')
    ax2.set_ylabel('SCC', fontsize=12)
    ax2.set_yticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    ax2.yaxis.set_ticklabels(['0.55', '0.60', '0.65', '0.70', '0.75', '0.8'], fontsize=12)
    # ax2.set_xlim(0, 10000)
    # ax2.set_ylim(0.2, 0.7)

    ax1.legend([l1, l2, l3, l4], labels=['SI-GCN', 'GNN_30', 'GM_P', 'RM'], bbox_to_anchor=(1.35, 0.8),
               borderaxespad=0.1, ncol=1)
    ax2.legend([l5, l6, l7, l8], labels=['SI-GCN', 'GNN_30', 'GM_P', 'RM'], bbox_to_anchor=(1.35, 0.35),
               borderaxespad=0.1, ncol=1)

    plt.show()


def sampling_effect_four_metrics():
    RMSE = np.array([[27.122, 27.293, 27.498, 28.772, 28.869, 28.107, 27.311, 28.663, 27.937, 27.914],
                     [24.512, 25.531, 25.367, 25.322, 25.531, 25.367, 25.394, 25.242, 25.288, 25.296],
                     [20.516, 20.620, 20.854, 20.191, 21.055, 20.826, 20.546, 21.049, 20.893, 20.728],
                     [18.398, 18.505, 18.660, 18.560, 18.405, 18.419, 18.315, 18.556, 18.503, 18.583]])

    MAPE = np.array([[29.1, 29.1, 28.6, 28.1, 27.7, 28.9, 27.9, 28.1, 27.8, 28.1],
                     [25.4, 25.3, 24.8, 25.1, 25.3, 24.8, 25.4, 25.6, 24.8, 26.4],
                     [23.4, 23.1, 22.5, 22.6, 22.6, 22.9, 23.6, 22.5, 23.7, 23.0],
                     [22.8, 21.6, 22.6, 22.3, 22.1, 22.8, 22.8, 22.5, 23.4, 23.2]])

    SCC = np.array([[0.558, 0.555, 0.549, 0.526, 0.530, 0.534, 0.550, 0.526, 0.546, 0.546],
                    [0.636, 0.628, 0.639, 0.625, 0.628, 0.639, 0.635, 0.638, 0.631, 0.632],
                    [0.709, 0.708, 0.716, 0.715, 0.713, 0.701, 0.703, 0.702, 0.712, 0.709],
                    [0.718, 0.722, 0.723, 0.725, 0.728, 0.719, 0.718, 0.719, 0.720, 0.727]])

    CPC = np.array([[0.846, 0.845, 0.844, 0.839, 0.839, 0.841, 0.844, 0.838, 0.842, 0.842],
                    [0.866, 0.863, 0.863, 0.863, 0.863, 0.863, 0.863, 0.863, 0.863, 0.864],
                    [0.884, 0.883, 0.884, 0.886, 0.884, 0.882, 0.883, 0.882, 0.882, 0.883],
                    [0.888, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.888, 0.887, 0.888]])

    RMSE = np.mean(RMSE, axis=1)
    MAPE = np.mean(MAPE, axis=1)
    SCC = np.mean(SCC, axis=1)
    CPC = np.mean(CPC, axis=1)

    xs = [1, 2, 3, 4]

    lw = 0.8
    ms = 6
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()
    l1, = ax1.plot(xs, RMSE, '--', marker='<', markersize=ms, color=colors[0], linewidth=lw, alpha=0.5, label='RMSE')
    l2, = ax1.plot(xs, MAPE, '--', marker='s', markersize=ms, color=colors[1], linewidth=lw, alpha=0.5, label='MAPE')
    l3, = ax1.plot(0, 0, '--', marker='D', markersize=ms, color=colors[2], linewidth=lw, alpha=0.5, label='SCC')
    l4, = ax1.plot(0, 0, '--', marker='o', markersize=ms, color=colors[3], linewidth=lw, alpha=0.5, label='CPC')
    ax1.set_ylabel('RMSE & MAPE(%)', fontname = 'Arial')
    ax1.set_ylim(18, 30)
    ax1.set_xlabel('Training set size', fontname='Arial')
    ax2 = ax1.twinx()
    l3, = ax2.plot(xs, SCC, '--', marker='D', markersize=ms, color=colors[2], linewidth=lw, alpha=0.5, label='SCC')
    l4, = ax2.plot(xs, CPC, '--', marker='o', markersize=ms, color=colors[3], linewidth=lw, alpha=0.5, label='CPC')

    ax2.set_ylabel('SCC & CPC', fontname = 'Arial')
    ax2.set_xticks(xs)
    ax2.xaxis.set_ticklabels(['20%', '40%', '60%', '80%'])
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax2.set_ylim(0.5, 0.9)
    ax2.set_xlim(0.9, 4.1)

    ax1.legend([l1, l2, l3, l4], labels=['RMSE', 'MAPE', 'SCC', 'CPC'], bbox_to_anchor=(0.22, 0.61), borderaxespad=0.1, ncol=1)
    #.legend([l3, l4], labels=['SCC', 'CPC'], bbox_to_anchor=(0.2, 0.55), borderaxespad=0.1, ncol=1)

    plt.show()


def training_set_size():
    RMSE = np.array([[27.122, 27.293, 27.498, 28.772, 28.869, 28.107, 27.311, 28.663, 27.937, 27.914],
                     [24.512, 25.531, 25.367, 25.322, 25.531, 25.367, 25.394, 25.242, 25.288, 25.296],
                     [20.516, 20.620, 20.854, 20.191, 21.055, 20.826, 20.546, 21.049, 20.893, 20.728],
                     [18.398, 18.505, 18.660, 18.560, 18.405, 18.419, 18.315, 18.556, 18.503, 18.583]])

    MAPE = np.array([[29.1, 29.1, 28.6, 28.1, 27.7, 28.9, 27.9, 28.1, 27.8, 28.1],
                     [25.4, 25.3, 24.8, 25.1, 25.3, 24.8, 25.4, 25.6, 24.8, 26.4],
                     [23.4, 23.1, 22.5, 22.6, 22.6, 22.9, 23.6, 22.5, 23.7, 23.0],
                     [22.8, 21.6, 22.6, 22.3, 22.1, 22.8, 22.8, 22.5, 23.4, 23.2]])

    SCC = np.array([[0.558, 0.555, 0.549, 0.526, 0.530, 0.534, 0.550, 0.526, 0.546, 0.546],
                    [0.636, 0.628, 0.639, 0.625, 0.628, 0.639, 0.635, 0.638, 0.631, 0.632],
                    [0.709, 0.708, 0.716, 0.715, 0.713, 0.701, 0.703, 0.702, 0.712, 0.709],
                    [0.718, 0.722, 0.723, 0.725, 0.728, 0.719, 0.718, 0.719, 0.720, 0.727]])

    CPC = np.array([[0.846, 0.845, 0.844, 0.839, 0.839, 0.841, 0.844, 0.838, 0.842, 0.842],
                    [0.866, 0.863, 0.863, 0.863, 0.863, 0.863, 0.863, 0.863, 0.863, 0.864],
                    [0.884, 0.883, 0.884, 0.886, 0.884, 0.882, 0.883, 0.882, 0.882, 0.883],
                    [0.888, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.888, 0.887, 0.888]])

    RMSE_mean = np.mean(RMSE, axis=1)
    MAPE_mean = np.mean(MAPE, axis=1)
    SCC_mean = np.mean(SCC, axis=1)
    CPC_mean = np.mean(CPC, axis=1)

    RMSE_err = np.std(RMSE, axis=1)
    SCC_err = np.std(SCC, axis=1)

    xs = [1, 2, 3, 4]

    lw = 0.6
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()
    l1 = ax1.errorbar(xs, RMSE_mean, yerr=RMSE_err, ecolor=colors[0], elinewidth=1, linewidth=lw, linestyle='--',
                 color=colors[0], capsize=2)
    l3 = ax1.errorbar(xs, SCC_mean, yerr=SCC_err, ecolor=colors[2], elinewidth=1, linewidth=lw, linestyle='--',
                      color=colors[2], capsize=2)
    ax1.set_ylabel('RMSE', fontname = 'Arial')
    ax1.set_ylim(18, 29)
    ax1.set_xlabel('Training set size', fontname='Arial')
    ax2 = ax1.twinx()
    l3 = ax2.errorbar(xs, SCC_mean, yerr=SCC_err, ecolor=colors[2], elinewidth=1, linewidth=lw, linestyle='--',
                 color=colors[2], capsize=2)
    ax2.set_ylabel('SCC', fontname = 'Arial')
    ax2.set_xticks(xs)
    ax2.xaxis.set_ticklabels(['20%', '40%', '60%', '80%'])
    ax2.set_yticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    ax2.set_ylim(0.5, 0.75)
    ax2.set_xlim(0.9, 4.1)

    ax1.legend([l1, l3], labels=['RMSE', 'SCC'], bbox_to_anchor=(0.22, 0.61), borderaxespad=0.1, ncol=1)

    plt.show()


# compute natural breaks of data
def fisher_jenks(d, cNum):
    X = np.array(d).reshape((-1, 1))
    fj = ps.esda.mapclassify.Fisher_Jenks(X, cNum)
    meanV = []
    for i in range(cNum):
        meanV.append(np.mean(X[np.where(i == fj.yb)]))
    return fj.bins, meanV


def negative_sampling_rate_RMSE():
    #RMSE = np.array([21.213, 20.159, 19.681, 19.744, 19.867, 19.892, 20.007, 20.059, 20.403, 20.595, 20.729])
    RMSE = np.array([21.155, 20.083, 19.736, 19.744, 19.792, 19.892, 20.007, 20.059, 20.366, 20.532, 20.62])
    print(np.min(RMSE), np.max(RMSE))
    SCC = np.array([0.659, 0.678, 0.693, 0.698, 0])

    xs = range(0, 11)
    lw = 0.6
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()
    ax1.plot(xs, RMSE, linewidth=lw, linestyle='-', marker='^', color=colors[3])
    ax1.set_ylabel('RMSE', fontname='Arial')
    ax1.set_ylim(19.7, 21.2)
    ax1.set_xticks(xs)
    ax1.xaxis.set_ticklabels(xs)
    ax1.set_xlabel('Negative sampling rate', fontname='Arial')

    #ax1.legend([l1], labels=['RMSE', 'SCC'], bbox_to_anchor=(0.22, 0.61), borderaxespad=0.1, ncol=1)

    plt.show()


def negative_sampling_rate():
    #RMSE = np.array([21.213, 20.159, 19.681, 19.744, 19.867, 19.892, 20.007, 20.059, 20.403, 20.595, 20.729])
    RMSE = np.array([21.155, 20.083, 19.736, 19.744, 19.792, 19.892, 20.007, 20.059, 20.366, 20.532, 20.62])
    print(np.min(RMSE), np.max(RMSE))
    SCC = np.array([0.661, 0.687, 0.702, 0.698, 0.702, 0.702, 0.701, 0.706, 0.703, 0.700, 0.706])
    print(np.min(SCC), np.max(SCC))

    xs = range(0, 11)
    lw = 0.6
    colors = ['orangered', 'hotpink', 'limegreen', 'skyblue']
    fig, ax1 = plt.subplots()
    l1, = ax1.plot(xs, RMSE, linewidth=lw, linestyle='-', marker='^', color=colors[3], label='RMSE')
    ax1.set_ylabel('RMSE', fontname='Arial')
    ax1.set_ylim(19.7, 21.2)
    ax1.set_xticks(xs)
    ax1.xaxis.set_ticklabels(xs)
    ax1.set_xlabel('Negative sampling rate', fontname='Arial')
    ax2 = ax1.twinx()
    l2, = ax2.plot(xs, SCC, linewidth=lw, linestyle='-', marker='.', color=colors[2], label='SCC')
    #ax1.legend([l1], labels=['RMSE', 'SCC'], bbox_to_anchor=(0.22, 0.61), borderaxespad=0.1, ncol=1)
    plt.legend([l1, l2], ['RMSE', 'SCC'], loc='lower right')
    plt.show()


def scatter_check():
    y = np.loadtxt('data/output_SI-GCN/output neg_20%/iter_32000.txt', dtype=np.float16)
    y = np.loadtxt('data/output_SI-GCN/output neg_80%/iter_37500.txt', dtype=np.float16)

    x = np.loadtxt('SI-GCN/data/taxi_th30/test.txt', dtype=np.uint32, delimiter='\t')[:, 3]

    p1 = plt.scatter(x, y, marker='.', color='green', s=10)

    #plt.xlim(0, 10)
    #plt.ylim(0, 10)
    #plt.xticks([0, 2, 4, 6, 8, 10])

    plt.xlabel('Time')
    plt.ylabel('Number')

    plt.show()



if __name__ == '__main__':
    #iter_rmse_scc()
    #check()
    #var_intensity()
    #var_distance()
    #limited_attributes()
    #var_threshold()
    #training_set_size()
    negative_sampling_rate()
    #scatter_check()


