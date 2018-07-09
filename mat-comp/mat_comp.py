from soft_impute import SoftImpute
import numpy as np
import matplotlib.pyplot as plt


def read_interaction_matrix(filename, gridnum):
    im = np.zeros((gridnum,gridnum), dtype=np.float64)
    with open(filename, 'r') as f:
        f.readline()
        line1 = f.readline().strip()
        while line1:
            sl1 = line1.split(',')
            line2 = f.readline().strip()
            sl2 = line2.split(',')
            if int(sl1[1]) == 1 and int(sl2[1]) == 1:
                im[int(sl1[-1]), int(sl2[-1])] += 1
            line1 = f.readline().strip()
    return im


if __name__ == '__main__':
    datafile = '../data/sj_051317.csv'
    im = read_interaction_matrix(datafile, 900)
    im_norm = (im-np.mean(im))/np.std(im)
    im_incomp = im_norm.copy()


x = [0,0,1,2,3,3]
y = [0,3,3,1,2,4]
incom_p[x, y] = np.nan
#com_p = SoftImpute().complete(incom_p)
#print(norm_p)
#print(incom_p)
#print(com_p)


e = []
xx = []
incom_f = norm_f.copy()
for i in range(100):
    x = np.random.randint(0, 25, 10, dtype=np.int)
    y = np.random.randint(0, 25, 10, dtype=np.int)
    incom_f[x, y] = np.nan
    com_f = SoftImpute().complete(incom_f)
    if i == 50 or i == 30:
        print(f[x, y])
        print(com_f[x, y]*np.std(f)+np.mean(f))
    mse = (((abs(norm_f[x, y]-com_f[x, y]))*np.std(f)+np.mean(f)).mean())
    xx.append(i)
    e.append(mse)

e.sort()
p1 = plt.scatter(xx, e, marker='.', color='green', s=50)

plt.xlim(0, 100)
plt.ylim(0, 100)
#plt.xticks([0, 2, 4, 6, 8, 10])

plt.xlabel('#')
plt.ylabel('mean error')

plt.show()
