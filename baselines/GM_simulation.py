'''
gravity model with simulated data and noise: illustration
'''


import numpy as np
from func import dis
import matplotlib.pyplot as plt
import seaborn as sns


class Region:
    def __init__(self, name, x, y, a, p):
        self.name = name
        self.x = x
        self.y = y
        self.a = a
        self.p = p


def gravity_model(A, B, beta, K):
    d = dis(A.x, A.y, B.x, B.y)
    return K * A.p * B.a / d**beta


def region_init():
    A = Region('A', 1, 12, 10, 10)
    B = Region('B', 13, 12, 15, 10)
    C = Region('C', 6, 10, 15, 20)
    D = Region('D', 0, 6, 15, 25)
    E = Region('E', 8, 6, 30, 40)
    F = Region('F', 15, 8, 40, 20)
    G = Region('G', 10, 1, 10, 20)

    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G}


def plot(flows, regions, beta, K):
    y = []
    x = []
    for f in flows:
        g = gravity_model(regions[f[0]], regions[f[1]], beta, K)

        y.append(np.log(g / (regions[f[0]].p * regions[f[1]].a)))
        x.append(np.log(dis(regions[f[0]].x, regions[f[0]].y, regions[f[1]].x, regions[f[1]].y)))
    x = np.array(x)
    y = np.array(y)
    print(x)
    font1 = {'family': 'Arial', 'weight': 'normal', 'size': 14}
    font2 = {'family': 'Arial', 'weight': 'normal', 'size': 14}
    sns.set_style('whitegrid')
    m, b = np.polyfit(x, y, 1)
    l1 = plt.scatter(x, y, 18, 'limegreen', label='ideal')
    lx = np.arange(1.4,2.9,0.1)
    plt.plot(lx, m * lx + b, linewidth=1, color='limegreen')

    #ny = np.random.normal(0, 0.1, y.shape[0]) + y + 0.1
    #print(ny)
    ny = np.array([-2.74954513,-1.98675937,-2.81374532,-2.22458175,-2.03047404,-1.38728483,-1.94625147,
                    -1.66228986,-2.00819296,-2.55600734]) + 0.1
    l2 = plt.scatter(x, ny, 18, 'orangered', label='observed')
    nm, nb = np.polyfit(x, ny, 1)
    plt.plot(lx, nm * lx + nb, linewidth=1, color='orangered')
    plt.xlim(1.4, 2.8)
    plt.ylim(-3, -1)
    x_ticks = np.arange(1.4, 3, 0.2)
    y_ticks = np.arange(-3, -0.5, 0.5)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.xlabel(r'$\log\mathit{d}$', font1)
    plt.ylabel(r'$\log\frac{\mathit{f}}{\mathit{pa}}$', font1)
    plt.legend(loc='upper right', prop=font2)
    plt.show()


if __name__ == '__main__':
    regions = region_init()
    flows = [('A', 'F'), ('B', 'E'), ('D', 'F'), ('E', 'A'), ('E', 'B'), ('E', 'C'), ('E', 'D'), ('E', 'G'),
             ('F', 'G'), ('G', 'D')]
    beta = 1
    K = 1
    for f in flows:
        g = int(gravity_model(regions[f[0]], regions[f[1]], beta, K))
        print(f[0], f[1], g, g/200)

    plot(flows, regions, beta, K)