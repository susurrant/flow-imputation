

import networkx as nx


def construct_net_from_file(filename, minSpeed = 2, maxSpeed = 150):
    si = {}
    nodes = set()
    with open(filename, 'r') as f:
        f.readline()
        while True:
            line1 = f.readline().strip()
            if line1:
                sl1 = line1.split(',')
                sl2 = f.readline().strip().split(',')
                if sl1[1] == '1' and minSpeed < float(sl1[-2]) < maxSpeed:
                    ogid = int(sl1[-1])
                    dgid = int(sl2[-1])
                    nodes.add(ogid)
                    nodes.add(dgid)
                    if (ogid, dgid) not in si:
                        si[(ogid, dgid)] = 0
                    si[(ogid, dgid)] += 1
            else:
                break

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for k, v in si.items():
        G.add_weighted_edges_from((k[0], k[1], v))

    return G