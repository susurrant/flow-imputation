# 统计SI-GCN结果

import numpy as np
import os
from scipy import stats


r = np.loadtxt('SI-GCN/data/taxi/test.txt', dtype=np.uint32, delimiter='\t')[:, 3]
path = 'data/output_SI-GCN/output/'
files = os.listdir(path)
for filename in files:
    p = np.loadtxt(path + filename, delimiter=',')
    smc = stats.spearmanr(r, p)
    scc = round(smc[0], 3)
    rmse = round(np.sqrt(np.mean(np.square(r - p))), 3)

    c = 0
    mape = 0
    for i in range(p.shape[0]):
        if r[i]:
            mape += np.abs((r[i] - p[i]) / r[i])
            c += 1

    stack = np.column_stack((p, r))
    cpc = round(2 * np.sum(np.min(stack, axis=1)) / np.sum(stack), 3)

    print('------------------------------------')
    print(filename)
    print('scc=', scc, 'cpc=', cpc, 'rmse=', rmse, 'mape=', round(mape / c, 3))
