
import numpy as np
from func import dis


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
    A = Region('A', 2, 6, 10, 10)
    B = Region('B', 6, 6, 15, 10)
    C = Region('C', 3, 5, 15, 20)
    D = Region('D', 1, 3, 15, 25)
    E = Region('E', 4, 3, 30, 40)
    F = Region('F', 7, 4, 40, 20)
    G = Region('G', 5, 1, 10, 20)

    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G}

if __name__ == '__main__':
    regions = region_init()
    flows = [('A', 'F'), ('B', 'E'), ('D', 'F'), ('E', 'A'), ('E', 'B'), ('E', 'C'), ('E', 'D'), ('E', 'G'),
             ('F', 'G'), ('G', 'D')]
    beta = 1
    K = 1
    for f in flows:
        g = int(gravity_model(regions[f[0]], regions[f[1]], beta, K))
        print(f[0], f[1], g/200)