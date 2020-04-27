# Python version: 2.7
import arcpy
import random

def gen_flow(data_file, out_file):
    with open(out_file, 'wb') as rf:
        rf.write('ox,oy,dx,dy\r\n')
        with open(data_file, 'r') as f:
            f.readline()
            line = f.readline().strip()
            while line:
                d1 = line.split(',')
                d2 = f.readline().strip().split(',')
                if d1[1] == '1' and d2[1] == '1':
                    if random.random() <= 0.1:
                        rf.write(d1[4]+','+d1[5]+','+d2[4]+','+d2[5]+'\r\n')
                line = f.readline().strip()


def gen_SI(pt_file, flow_file, out_txt_file, out_shp_file):
    cen = {}
    with open(pt_file, 'r') as f:
        f.readline()
        line = f.readline().strip()
        while line:
            d = line.split(',')
            cen[d[1]] = (d[2], d[3])
            line = f.readline().strip()

    with open(out_txt_file, 'wb') as rf:
        rf.write('ox,oy,dx,dy,m\r\n')
        with open(flow_file, 'r') as f:
            f.readline()
            line = f.readline().strip()
            while line:
                d = line.split('\t')
                rf.write(cen[d[0]][0]+','+cen[d[0]][1]+','+cen[d[2]][0]+','+cen[d[2]][1]+','+d[3]+'\r\n')
                line = f.readline().strip()

    xy2line(out_txt_file.split('/')[-1], out_shp_file)


#XY To Line
def xy2line(input_table, out_lines):
    arcpy.env.workspace = './data'
    spRef = r"Coordinate Systems\Geographic Coordinate Systems\World\WGS 1984.prj"
    arcpy.XYToLine_management(input_table, out_lines, 'ox', 'oy', 'dx', 'dy', id_field='m', spatial_reference=spRef)


if __name__ == '__main__':
    pt_file = 'data/pt_fn_1km_cen.txt'
    flow_file = 'data/taxi_1km_t30.txt'
    out_file = 'data/flow_1km_t30.txt'
    out_shp_file = 'flow_1km_051317_t30'
    gen_SI(pt_file, flow_file, out_file, out_shp_file)