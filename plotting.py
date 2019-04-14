
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def iter_error():
    sns.set_style('darkgrid')
    gnn_rmse = np.loadtxt('GNN_RMSE.txt')/100
    gnn_smc = np.loadtxt('GNN_SMC.txt')
    print(min(gnn_rmse), max(gnn_rmse))
    x = np.arange(50, 10001, 50)
    plt.plot(x, gnn_rmse, color='red', linewidth=1)
    plt.plot(x, gnn_smc, color='blue', linewidth=1)
    plt.show()


if __name__ == '__main__':
    iter_error()