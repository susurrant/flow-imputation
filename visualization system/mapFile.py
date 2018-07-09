#!/usr/bin/env python3
# -*- coding:  utf-8 -*-

from grid import Grid


def readData(filename, gridnum):
    grids = {}
    for i in range(gridnum):
        grids[i] = Grid(i)

    with open(filename, 'r') as f:
        f.readline()
        line1 = f.readline().strip()
        while line1:
            sl1 = line1.split(',')
            line2 = f.readline().strip()
            sl2 = line2.split(',')
            if int(sl1[1]) == 1 and int(sl2[1]) == 1:
                grids[int(sl1[-1])].add_flow(int(sl2[-1]))

            line1 = f.readline().strip()
    return grids


def init_grids(grids, settings):
    rows, cols, grid_width = settings['rows'], settings['cols'], settings['grid_width']
    yoffset, xoffset, oy, ox = settings['yoffset'], settings['xoffset'], settings['oy'], settings['ox']
    for gid in grids:
        j = gid // cols
        i = gid % cols

        grids[gid].ceny = (rows - j - yoffset + 0.5) * grid_width + oy
        grids[gid].cenx = (i - xoffset + 0.5) * grid_width + ox

        top = (rows - j - yoffset) * grid_width + oy
        bottom = (rows - j - yoffset + 1) * grid_width + oy
        left = (i - xoffset) * grid_width + ox
        right = (i - xoffset + 1) * grid_width + ox
        grids[gid].bbox = [left, top, right, bottom]


def relate2data(filename, settings):
    grids = readData(filename, settings['rows']*settings['cols'])
    init_grids(grids, settings)
    return grids