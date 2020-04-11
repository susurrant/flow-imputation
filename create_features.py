
import csv

with open('data/fn_1km.csv', 'w', newline='') as rf:
    sheet = csv.writer(rf, delimiter='\t')
    #sheet.writerow(['ogid', 'dgid', 'm'])
    with open('data/pt_fn_vertices.txt', 'r') as f:
        f.readline()
        i = 1
        line = f.readline().strip()
        while line:
            line = f.
