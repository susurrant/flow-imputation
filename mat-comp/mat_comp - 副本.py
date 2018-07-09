from soft_impute import SoftImpute
import numpy as np
import matplotlib.pyplot as plt


p = np.array([[10, 20, 0, 2, 0], [15, 20, 2, 8, 4], [4, 5, 6, 2, 0], [0, 4, 6, 30, 15], [0, 1, 2, 20, 10]])

f = np.zeros((25, 25))

for i in range(25):
    r1 = i // 5
    c1 = i % 5
    for j in range(25):
        if i == j:
            f[i, j] = p[r1, c1]
            continue
        r2 = j // 5
        c2 = j % 5
        f[i, j] = p[r1, c1]*p[r2, c2]/(abs(r1-r2)+abs(c1-c2))


norm_p = (p-np.mean(p))/np.std(p)
norm_f = (f-np.mean(f))/np.std(f)

incom_p = norm_p.copy()
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
