
import numpy as np
import random

def norm():
    features = np.loadtxt('features_raw.txt', dtype=np.float, delimiter='\t')
    features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    np.savetxt('features.txt', features, fmt='%.3f', delimiter='\t')


def split():
    p_data = []
    with open('data.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            p_data.append(line)

    t = set(range(len(p_data)))
    train_set = set(random.sample(range(len(p_data)), int(0.6 * len(p_data))))
    s = t - train_set
    test_set = set(random.sample(s, int(0.2 * len(p_data))))
    valid_set = s - test_set

    with open('train.txt', 'w', newline='') as f:
        for i in train_set:
            f.write(p_data[i]+'\r\n')
    with open('test.txt', 'w', newline='') as f:
        for i in test_set:
            f.write(p_data[i]+'\r\n')
    with open('valid.txt', 'w', newline='') as f:
        for i in valid_set:
            f.write(p_data[i]+'\r\n')

if __name__ == '__main__':
    norm()
    split()