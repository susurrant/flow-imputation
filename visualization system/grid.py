# -*- coding: utf-8 -*-：

class Grid(object):
    def __init__(self, gid):
        # 格网ID
        self.gid = gid

        # 坐标
        self.cenx = -1
        self.ceny = -1
        self.bbox = []

        # 关联交互集合
        self.SI = {}

    def add_flow(self, gid):
        if gid not in self.SI:
            self.SI[gid] = 0
        self.SI[gid] += 1

