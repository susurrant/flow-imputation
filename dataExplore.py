import numpy as np
import csv

def process():
    flows = {}
    grids = {}
    with open('data/unicom_OD.csv', 'r') as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.strip().split(',')
            if d[0] in ['20170109', '20170110', '20170111', '20170112', '20170113']:
                if (d[2], d[3]) not in flows:
                    flows[(d[2], d[3])] = 0
                flows[(d[2], d[3])] += int(d[4])
                if d[2] not in grids:
                    grids[d[2]] = 0
                if d[3] not in grids:
                    grids[d[3]] = 0
            line = f.readline()

    md = {}
    for g, v in flows.items():
        if v not in md:
            md[v] = 0
        md[v] += 1
        if v > 100:
            grids[g[0]] = 1
            grids[g[1]] = 1
    #print(md)

    with open('data/unicom_grid.csv', 'w', newline='') as f:
        sheet = csv.writer(f)
        sheet.writerow(['grid', 'tag'])
        for grid, tag in grids.items():
            sheet.writerow([grid, tag])

if __name__ == '__main__':
    process()