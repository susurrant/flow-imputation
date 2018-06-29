# -*- coding: utf-8 -*-：


def read_data(filename):
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line:
            sl = line.split(',')
