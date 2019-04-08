

import tensorflow as tf
import numpy as np
from func import *


# 构造添加神经层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def read_data(path, normalization=False):
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    tr_f = read_flows(path + 'train.txt')
    te_f = read_flows(path + 'test.txt')
    if normalization:
        features = read_features(path + 'entities.dict', path + 'features.txt')
    else:
        features = read_features(path + 'entities.dict', path + 'features_raw.txt')

    for k in tr_f:
        if k[2]:
            train_y.append(k[2])
            train_X.append([features[k[0]][3], features[k[1]][2],
                            dis(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1])])

    for k in te_f:
        if k[2]:
            test_y.append(k[2])
            test_X.append([features[k[0]][3], features[k[1]][2],
                           dis(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1])])

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


if __name__ == '__main__':
    path = 'SI-GCN/data/taxi/'
    train_X, train_y, test_X, test_y = read_data(path, normalization=True)

    # 定义占位符
    xs = tf.placeholder(tf.float32, [None, 3])
    ys = tf.placeholder(tf.float32, [None, 1])

    # 假设一个输入层，十个隐藏层，一个输出层
    l1 = add_layer(xs, 3, 20, activation_function=tf.nn.relu)

    # 定义输出层
    prediction = add_layer(l1, 20, 1, activation_function=None)

    # 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # Session
    sess = tf.Session()
    sess.run(init)

    # 机器开始学习
    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
