
import numpy as np
import random

def norm():
    features = np.loadtxt('features_raw.txt', dtype=np.float32, delimiter='\t')
    features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    np.savetxt('features.txt', features, fmt='%.3f', delimiter='\t')


def split():
    p_data = np.loadtxt('data.txt', dtype=np.uint32, delimiter='\t')
    #p_data[:, 3] = np.log(p_data[:, 3])

    t = set(range(len(p_data)))
    train_set = set(random.sample(range(len(p_data)), int(0.6 * len(p_data))))
    train_data = p_data[list(train_set)]
    s = t - train_set
    test_set = set(random.sample(s, int(0.2 * len(p_data))))
    test_data = p_data[list(test_set)]
    valid_set = s - test_set
    valid_data = p_data[list(valid_set)]

    np.random.shuffle(train_data)
    np.savetxt('train.txt', train_data, fmt='%d', delimiter='\t')
    np.savetxt('test.txt', test_data, fmt='%d', delimiter='\t')
    np.savetxt('valid.txt', valid_data, fmt='%d', delimiter='\t')

if __name__ == '__main__':
    norm()
    split()