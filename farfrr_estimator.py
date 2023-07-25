import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



def calc_farfrr(match, other, thres = np.linspace(0,1,1001)):
    # thres = np.linspace(0,1,1001)
    far, frr = [], []
    for thre in thres:
        far.append(np.sum(other<thre))
        frr.append(np.sum(match>thre))

    far = np.array(far)/len(other)
    frr = np.array(frr)/len(match)

    Data = pd.DataFrame([])
    Data.index = thres
    Data["FAR"] = far
    Data["FRR"] = frr
    Data["det"] = np.abs(Data['FAR']-Data['FRR'])
    return Data

def getFARandFRRData(gens,imps,mode="osiris",ths = np.arange(0,1010,1)/1000):

    compFRRVals = lambda a,b : a<b if mode=='osiris' else a>b
    compFARVals = lambda a,b : a>b if mode=='osiris' else a<b
    rev = False
    Data = pd.DataFrame([])
    Data.index = ths
    if mode == "osiris":
        rev = True
        ths = ths[::-1].copy()
    
    gens = np.array(gens)
    imps = np.array(imps)

    gens = np.sort(gens)
    imps = np.sort(imps)
    if rev:
        gens = gens[::-1]
    else:
        imps = imps[::-1]
    

    FRR = []
    curIdx = 0
    for thr in ths:
        th = thr
        while (curIdx<len(gens)):
            if compFRRVals(gens[curIdx],th):
                FRR.append([th,max(curIdx,0)/len(gens)])
                break
            curIdx+=1

    FAR = []
    curIdx = 0
    for th in ths[::-1]:
        while (curIdx<len(imps)):
            if compFARVals(imps[curIdx],th):
                FAR.append([th,max(curIdx,0)/len(imps)])
                break
            curIdx+=1

    FAR = np.array(FAR).T
    FRR = np.array(FRR).T
    Data.loc[FAR[0],'FAR'] = FAR[1]
    Data.loc[FRR[0],'FRR'] = FRR[1]

    Data["det"] = np.abs(Data['FAR']-Data['FRR'])

    return Data



def getFARandFRRData2(gens,imps,mode="osiris",ths = np.arange(0,1010,1)/1000):

    rev = False    
    Data = pd.DataFrame([])
    Data.index = ths
    if mode == "osiris":
        rev = True
        ths = ths[::-1].copy()
    gens = np.sort(gens)
    imps = np.sort(imps)

    FRR = []
    gensl = len(gens)
    curIdx = gensl if rev else 0
    for th in ths:
        idx = np.searchsorted(gens,th, side = "right" if rev else "left")
        if rev:
            curIdx = idx
            FRR.append([th,(gensl-curIdx)/gensl])
            gens = gens[:idx]
        else:
            curIdx += idx
            FRR.append([th,curIdx/gensl])
            gens = gens[idx:]

    FAR = []
    impsl = len(imps)
    curIdx = 0 if rev else len(imps)
    for th in ths[::-1]:
        idx = np.searchsorted(imps,th, side = "left" if rev else "right")
        if rev:
            curIdx += idx
            FAR.append([th,curIdx/impsl])
            imps = imps[idx:] 
        else:
            curIdx = idx
            FAR.append([th,(impsl-curIdx)/impsl])
            imps = imps[:idx]

    FAR = np.array(FAR).T
    FRR = np.array(FRR).T[:,::-1]

    
    minIdx = np.abs(FAR[1]-FRR[1]).argmin()
    
    eerth = FAR[0,minIdx]
    eer = (FAR[1,minIdx] + FRR[1,minIdx])/2

    
    Data.loc[FAR[0],'FAR'] = FAR[1]
    Data.loc[FRR[0],'FRR'] = FRR[1]
    Data["det"] = np.abs(Data['FAR']-Data['FRR'])

    return Data

import time

datalength = 5000
scMat = np.random.rand(datalength,datalength)
classes = 10
arrs = np.repeat(np.arange(classes),datalength/classes,axis = 0)
xl = np.repeat(arrs.reshape(1,len(arrs)), len(arrs), axis=0)
yl = np.repeat(arrs.reshape(len(arrs),1), len(arrs), axis=1)
maskG = (xl==yl)

trimask = np.tril(np.ones_like(maskG),k=-1)
gens = scMat[(maskG*trimask)>0]
imps = scMat[(~maskG*trimask)>0]


numOfIter = 1
time.sleep(3)

for det in [1000,100,10]:
    thres = np.arange(0,10010,det)/10000

    st = time.perf_counter()
    for i in range(numOfIter):
        my = getFARandFRRData(gens,imps,mode = "osiris",ths = thres)
        minIdx = my["det"].argmin()
        minsc = my.index[minIdx]
        eer = (my["FAR"].iloc[minIdx]+my["FRR"].iloc[minIdx])/2
        print(f"sort1: eer = {eer:0.3f} at score = {minsc:0.3f}")
    print(f"{det/10000}:  using sort1 {(time.perf_counter()-st)/numOfIter:0.3f} sec")

    st = time.perf_counter()
    for i in range(numOfIter):
        my = getFARandFRRData2(gens,imps,mode = "osiris",ths = thres)
        minIdx = my["det"].argmin()
        minsc = my.index[minIdx]
        eer = (my["FAR"].iloc[minIdx]+my["FRR"].iloc[minIdx])/2
        print(f"sort2: eer = {eer:0.3f} at score = {minsc:0.3f}")
    print(f"{det/10000}:  using sort2 {(time.perf_counter()-st)/numOfIter:0.3f} sec")
    
    st = time.perf_counter()
    for i in range(numOfIter):
        ot = calc_farfrr(gens,imps,thres = thres)
        minIdx = ot["det"].argmin()
        minsc = ot.index[minIdx]
        eer = (ot["FAR"].iloc[minIdx]+ot["FRR"].iloc[minIdx])/2
        print(f"w/o sort: eer = {eer:0.3f} at score = {minsc:0.3f}")
    print(f"{det/10000}: not using sort {(time.perf_counter()-st)/numOfIter:0.3f} sec")
    print()
