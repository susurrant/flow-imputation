# -*- coding: utf-8 -*-

import random
import copy
import math
from func import *



# 数据预处理函数
def preprocessData(points, flows):
    PointName = []

    for rcd in flows:
        if rcd[0] not in PointName:
            PointName.append(rcd[0])
        if rcd[1] not in PointName:
            PointName.append(rcd[1])
            
        
    PointNum = len(PointName)
        
    #build a matrix for interactions
    #RT part: interaction strength
    #LB part: distance in km
        
    InterData = [[-1.0 for a in range(PointNum)] for b in range(PointNum)]
    lon1 = -1
    lat1 = -1
    lon2 = -1
    lat2 = -1
    #只计算有流存在的两个城市的距离
    for rcd in flows:
        n1 = PointName.index(rcd[0])
        for p in points:
            if p[0] == rcd[0]:
                lon1 = p[1]
                lat1 = p[2]
                break
        n2 = PointName.index(rcd[1])
        for p in points:
            if p[0] == rcd[1]:
                lon2 = p[1]
                lat2 = p[2]
                break

        if n1 > n2:
            t = n1
            n1 = n2
            n2 = t
        InterData[n1][n2] = float(rcd[2])
        InterData[n2][n1] = dis(lon1, lat1, lon2, lat2)
    ValidPair = len(flows)
    
    return InterData,PointNum,ValidPair,PointName


#创建流
def CreateFlows(CitySize, PointNum, beta, InterData):
    for i in range(0, PointNum):
        for j in range(0,i):
            if InterData[i][j] > 0:  #valid pair
                InterData[j][i] = CitySize[i]*CitySize[j]/pow(InterData[i][j],beta)
            else:
                InterData[j][i] = -1


#抽取流
def ExtractFlowData(InterData, ValidPair, PointNum):
    Data = [0.0]*ValidPair
    Count = 0
    for i in range(0, PointNum):
        for j in range(i+1, PointNum):
            if InterData[i][j] > 0.0:
                Data[Count] = InterData[i][j]
                Count += 1
    
    return Data


#计算一维皮尔森相关系数
def PearsonCoefficient1D(data1, data2, size):
    mean1 = 0.0
    mean2 = 0.0
    i = 0
        
    while i < size:
        mean1 += data1[i]
        mean2 += data2[i]
        i += 1

    mean1 /= size
    mean2 /= size
    cov1 = 0.0
    cov2 = 0.0
    cov12 = 0.0
    
    i = 0
    while i < size:
        try:
            cov12 += (data1[i]-mean1)*(data2[i]-mean2)
            cov1 += (data1[i]-mean1)*(data1[i]-mean1)
            cov2 += (data2[i]-mean2)*(data2[i]-mean2)
            i += 1
        except:
            print(i)

    if abs(cov1)<0.00000001 or abs(cov2)<0.00000001: return 0
    return cov12/math.sqrt(cov1)/math.sqrt(cov2)


#粒子群搜索
def PSOSearch(InterData, PointNum, ValidPair, InitialSizes, beta, ParticleNum, SearchRange, w, c1, c2):
    Particles = [[0.0 for a in range(PointNum)] for b in range(ParticleNum)]

    for i in range(0, ParticleNum):
        for j in range(0, PointNum):
            if i == 0:
                Particles[i][j] = InitialSizes[j]
            else:
                Particles[i][j] = InitialSizes[j]+ (random.random()*SearchRange/5-SearchRange/10)
            if Particles[i][j] > SearchRange: Particles[i][j] = SearchRange
            if Particles[i][j] < 0: Particles[i][j]= 0
    
    #print Particles 
    Velocity = [[random.random()*SearchRange - SearchRange/2 for a in range(PointNum)] for b in range(ParticleNum)]
    #print Velocity
    
    
    gBestParticleScore = 0.0
    gBestParticle = [0.0]*PointNum

    pBestParticleScore = [0.0]*ParticleNum
    pBestParticle = [[0.0 for a in range(PointNum)] for b in range(ParticleNum)]

    RealFlowData = ExtractFlowData(InterData,ValidPair, PointNum)
    
    InterDataTemp = copy.deepcopy(InterData)
    IterCount = 0
    while 1:
        tBestScore = 0
        for i in range(0,ParticleNum):
            CreateFlows(Particles[i], PointNum, beta, InterDataTemp)
            FitData = ExtractFlowData(InterDataTemp, ValidPair, PointNum)

            gof = PearsonCoefficient1D(FitData, RealFlowData, ValidPair)
            if gof > tBestScore:
                tBestScore = gof
            
            #print gof
            if gof > pBestParticleScore[i]:
                pBestParticleScore[i] = gof
                for j in range(0,PointNum):
                    pBestParticle[i][j] = Particles[i][j]
                
            if gof > gBestParticleScore:
                gBestParticleScore = gof
                for j in range(0,PointNum):
                    gBestParticle[j] = Particles[i][j]

        #update particles
        maxVelocity = 0
        for i in range(0,ParticleNum):
            nc1 = c1 *random.random()
            nc2 = c2 *random.random()
            for j in range(0,PointNum):
                newVelocity = Velocity[i][j]*w + nc1*(pBestParticle[i][j]-Particles[i][j]) + nc2 *(gBestParticle[j]-Particles[i][j])
                #print i,j,c1,c2,Velocity[i][j],newVelocity
                
                if newVelocity + Particles[i][j] > SearchRange:
                    newVelocity = SearchRange - Particles[i][j]
                if newVelocity + Particles[i][j] < 0:
                    newVelocity = - Particles[i][j]

                if abs(newVelocity) > maxVelocity: maxVelocity = newVelocity
                    
                Velocity[i][j] = newVelocity
                Particles[i][j] += newVelocity
        #print gBestParticleScore, tBestScore, maxVelocity
        IterCount += 1
        if IterCount >= 1000 or maxVelocity < 5 or gBestParticleScore > 0.98:
            break

    return gBestParticleScore, gBestParticle


def InitSize(InterData, PointNum):
    print('init size...')
    Size = [0.0] * PointNum
    for i in range(0, PointNum):
        for j in range(i+1, PointNum):
            if InterData[i][j] > 0.0:
                Size[i] += InterData[i][j]
                Size[j] += InterData[i][j]
    return Size


#主调函数
def gravityFit(points, flows):
    InterData, PointNum, ValidPair, PointName = preprocessData(points, flows)
    print('grid num:', PointNum)

    Sizes = InitSize(InterData, PointNum)
    maxSize = max(Sizes)
    for i in range(0, PointNum):
        Sizes[i] = Sizes[i] / maxSize * 1000

    bestScoreResult = 0
    estSizeResult = []
    bestBeta = 0.05
    print('start PSO...')
    for beta in range(50, 180, 5):
        bs, estSize = PSOSearch(InterData, PointNum, ValidPair, Sizes, beta/100, 50, 1000, 1, 2.0, 2.0)
        #print('  beta:', beta / 100, 'score:', bs)
        #for bp in estSize:
        #    print(bp)
        if bs > bestScoreResult:
            bestScoreResult = bs
            bestBeta = beta/100
            estSizeResult = copy.deepcopy(estSize)

    print('best beta:', bestBeta)
    result = {}
    for i in range(0, PointNum):
        result[PointName[i]] =  estSizeResult[i]
    return result


def read_data(path):
    features = read_features(path + 'entities.dict', path + 'features_raw.txt')
    pts = []
    for k, c in features.items():
        pts.append([k, c[0], c[1]])

    tr_f = read_flows(path + 'train.txt')
    return pts, tr_f


class Region:
    def __init__(self, name, x, y, a, p):
        self.name = name
        self.x = x
        self.y = y
        self.a = a
        self.p = p


def gravity_model(A, B, beta, K):
    d = dis(A.x, A.y, B.x, B.y)
    return K * (A.p+A.a) * (B.p+B.a) / d**beta


def region_init():
    A = Region('A', 1, 12, 10, 10)
    B = Region('B', 13, 12, 15, 10)
    C = Region('C', 6, 10, 15, 20)
    D = Region('D', 0, 6, 15, 25)
    E = Region('E', 8, 6, 30, 40)
    F = Region('F', 15, 8, 40, 20)
    G = Region('G', 10, 1, 10, 20)

    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G}


if __name__ == '__main__':
    path = '../SI-GCN/data/taxi/'
    '''
    points = []
    flows = []
    regions = region_init()
    for r in regions:
        points.append([regions[r].name, regions[r].x, regions[r].y])
        print(r, regions[r].p+regions[r].a)

    cflows = [('A', 'F'), ('B', 'E'), ('D', 'F'), ('E', 'A'), ('E', 'B'), ('E', 'C'), ('E', 'D'), ('E', 'G'),
             ('F', 'G'), ('G', 'D')]
    for f in cflows:
        g = int(gravity_model(regions[f[0]], regions[f[1]], 1, 1))
        flows.append([f[0], f[1], g])
    '''
    points, flows = read_data(path)

    res = gravityFit(points, flows)
    for r in res:
        print(r, res[r])
