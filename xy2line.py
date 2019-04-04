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


#XY To Line
def xy2line(input_table, out_lines):
    arcpy.env.workspace = './data'
    spRef = r"Coordinate Systems\Geographic Coordinate Systems\World\WGS 1984.prj"
    arcpy.XYToLine_management(input_table, out_lines, 'ox', 'oy', 'dx', 'dy', spatial_reference=spRef)


def spatialJoin(target_file, join_file):
    arcpy.env.workspace = './data'
    try:
        # spatial join
        arcpy.SpatialJoin_analysis(target_file, join_file, 'sj_flow_051317')

        # export joined data
        #arcpy.ExportXYv_stats('sj_lyr.shp', output_fields, 'COMMA', 'sj_' + target_file, 'ADD_FIELD_NAMES')
    except Exception as err:
        print(err.args[0])


if __name__ == '__main__':
    input_table = 'flow_051317.csv'
    out_lines = 'flow_051317'
    gen_flow('data/taxi_sj_1km_051317.txt', 'data/flow_051317.csv')
    xy2line(input_table, out_lines)
    #spatialJoin('tem_vg.shp', 'flow_051317.shp')
