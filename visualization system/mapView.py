#!/usr/bin/env python3
# -*- coding:  utf-8 -*-

"""map view
"""


from tkinter import *
from tkinter import messagebox, filedialog
from mapFile import relate2data
import numpy as np
from LL2UTM import LL2UTM_USGS

class MapGUI(Frame):
    def __init__(self, master):
        settings = self.init_settings()

        Frame.__init__(self, master, bg='#e9e7ef', height=settings['height'],
                       width=settings['width']+settings['pFrm_width'])

        datafile = '../data/sj_051317.csv'
        grids = relate2data(datafile, settings)
        self.threshold = 100

        maxnum = 0
        for ogid in grids:
            for v in grids[ogid].SI.values():
                if v > maxnum:
                    maxnum = v
        settings.update({'maxnum': maxnum})
        self.pFrm = ParameterFrm(self, settings)  # parameter view
        self.pFrm.place(x=1000, y=0)

        self.pc = pattern_canvas(self, grids, settings) # pattern map canvas
        self.pc.place(x=0, y=0)

        self.show()

    @staticmethod
    def init_settings():
        settings = dict()
        settings.update({'height': 1000, 'width': 1000, 'pFrm_width': 150,
                         'grid_width': 32, 'xoffset': 0, 'yoffset': 1, 'ox': 20, 'oy': 20, 'margin': 3,
                         'rows': 30, 'cols': 30, 'xoff': 431000, 'yoff': 4400300, 'trans_scale': 31})

        return settings

    def update_data(self):
        self.threshold = self.pFrm.threshold.get()
        self.show()

    def show(self):
        self.pc.invalidate()


class ParameterFrm(Frame):
    def __init__(self, master, settings):
        bgc = '#e9e7ef'
        Frame.__init__(self, master, bg=bgc, height=settings['height'], width=settings['pFrm_width'])

        #----------------------parameters------------------------
        self.threshold = DoubleVar()
        #--------------------------------------------------------

        Label(self, text='Threshold', bg=bgc).place(x=20, y=70)
        self.threshold.set(master.threshold)
        self.thScale = Scale(self, orient=VERTICAL, bg=bgc, length=200, from_=1, to=settings['maxnum'],
                             resolution=1, variable=self.threshold)
        self.thScale.place(x=50, y=100)

        self.appbtn = Button(self, text='Apply', bg=bgc, relief=RAISED, height=1, width=15, command=master.update_data)
        self.appbtn.place(x=20, y=850)

        self.expbtn = Button(self, text='Export', bg=bgc, relief=RAISED, height=1, width=15, command=self.save_img)
        self.expbtn.place(x=20, y=900)

    def save_img(self):
        save_filename = filedialog.asksaveasfilename(filetypes = (("jpg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))
        messagebox.showinfo('OK', 'successfully saved!')

    def show_Var(self):
        print(self.threshold.get())


class pattern_canvas(Canvas):
    def __init__(self, master, grids, settings):
        Canvas.__init__(self, master, height = settings['height'], width = settings['width'], bg = 'white')
        self.grids = grids
        self.settings = settings
        self.ringroad = []
        self.read_ringroad()

        def mouseLeftClick(event):
            mgid = -1
            mdis = float('inf')
            for gid in self.grids:
                dis = np.sqrt((self.grids[gid].cenx - event.x) ** 2 + (self.grids[gid].ceny - event.y) ** 2)
                if dis < mdis:
                    mgid = gid
                    mdis = dis
            if mdis > self.settings['gridWidth']:
                self.master.show()
                return
            self.highlight(mgid)

        self.bind("<Button>", mouseLeftClick)

    def read_ringroad(self):
        with open('../data/ringroad_pt.csv', 'r') as f:
            lines = f.readlines()
            tag = 0
            pts = []
            for line in lines[1:]:
                sl = line.strip().split(',')
                x, y = LL2UTM_USGS(float(sl[3]), float(sl[2]))
                x = (x - self.settings['xoff']) / self.settings['trans_scale']
                y = self.settings['height'] - (y - self.settings['yoff']) / self.settings['trans_scale']

                if int(sl[1]) == tag:
                    pts.append((x, y))
                else:
                    self.ringroad.append(pts)
                    tag = int(sl[1])
                    pts = [(x, y)]
            self.ringroad.append(pts)

    def draw_ring_roads(self):
        for pts in self.ringroad:
            self.create_line(pts, fill='#b25d25', width=1)

    def invalidate(self):
        self.delete(ALL)

        for gid in self.grids:
            self.create_rectangle(self.grids[gid].bbox, outline='#bce672')

        self.draw_ring_roads()

        for ogid in self.grids:
            for dgid in self.grids[ogid].SI:
                if self.grids[ogid].SI[dgid] >= self.master.threshold:
                    self.create_line([self.grids[ogid].cenx, self.grids[ogid].ceny, self.grids[dgid].cenx,
                                      self.grids[dgid].ceny], width=0.5, fill='#3d3b4f', arrow=LAST)

    def drawLegend(self):
        pass

    def highlight(self, mgid):
        pass
        #self.invalidate()
        #self.create_line(self.grids[mgid].border, width=3, fill='#0000ff')
