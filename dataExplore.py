import numpy as np
import csv

def unicom_data():
    grid_map = {}
    with open('data/250_is_500.csv', 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            d = line.strip().split(',')
            grid_map[d[1]] = d[2] # map 250 to 500
            line = f.readline()

    flows_250 = {}
    with open('data/unicom_OD.csv', 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            d = line.strip().split(',')
            if d[0] in ['20170109', '20170110', '20170111', '20170112', '20170113']:
                if (d[2], d[3]) not in flows_250:
                    flows_250[(d[2], d[3])] = 0
                flows_250[(d[2], d[3])] += int(d[4])
            line = f.readline()

    flows_500 = {}
    for g, m in flows_250.items():
        if g[0] in grid_map and g[1] in grid_map:
            k = (grid_map[g[0]], grid_map[g[1]])
            if k not in flows_500:
                flows_500[k] = 0
            flows_500[k] += m

    with open('data/unicom_500.csv', 'w', newline='') as rf:
        sheet = csv.writer(rf)
        sheet.writerow(['o', 'd', 'm'])
        for g, m in flows_500.items():
            sheet.writerow([g[0], g[1], m])


if __name__ == '__main__':
    unicom_data()