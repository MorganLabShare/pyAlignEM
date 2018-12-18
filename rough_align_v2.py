#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 18:53:45 2018

@author: karl
"""

import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageOps
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
#PIL thinks that big images are attacks
Image.MAX_IMAGE_PIXELS=1024000001

sliceNum=24

srcDir = "/media/karl/morganlab/MasterRaw/hxR/hxR_waf003_04nm_single_v3"
srcDir = "/home/karl/Data/fullResLocal"
dirList = os.listdir( srcDir )
dirList.sort()
sliceList = [s for s in dirList if 'waf' in s and 'Mont' in s]
#stackIDList = range(len(sliceList))
stackIDList = list(range(sliceNum))

dstDir = "/home/karl/Data/hxR_waf003_single_test"

rawImageList = list(range(sliceNum))
DSsize=(8000,8000)

featureList = list(range(sliceNum))

def loadImage(stackID):
    curSlice=sliceList[stackID]
    fileList=os.listdir( srcDir+"/"+str(curSlice))
    fileName=[s for s in fileList if len(s)<30 and '.tif' in s][0]
    imraw=Image.open( srcDir+"/"+curSlice+"/"+fileName )
    imDS=imraw.resize(DSsize,resample=Image.BILINEAR)
    imDSL=imDS.convert('L')
    imDSLinv=ImageOps.invert(imDSL)
    imDSinv=imDSLinv.convert('P')
    #rawImageList[stackID]=imDSinv
    return imDSinv

def featExtractAKAZE(image):
    akazeDetector = cv2.AKAZE_create()
    (kps, descs) = akazeDetector.detectAndCompute(np.array(image), None)
    img=cv2.drawKeypoints(np.array(image), kps, None, color = (255,0,0))
    return (kps, descs, img)

def featExtractORB(image):
    orbDetector = cv2.ORB_create()
    orbDetector.setMaxFeatures(2500)
    kps = orbDetector.detect(np.array(image), None)
    (kps, descs) = orbDetector.compute(np.array(image), kps)
    img=cv2.drawKeypoints(np.array(image), kps, None, color = (255,0,0))
    return (kps, descs, img)

def shimage(image):
    img = Image.fromarray(image)
    img.show()

def displacementFinder(goodMatches,pointsA,pointsB):
    dists=[]
    for match in range(len(goodMatches)):
        locA=pointsA[0]
        locB=pointsB[0]
        diffEuc=locB-locA
        dists.append(diffEuc)
    return dists    

def removeOutliers2D(ptsX, ptsY, stdevCutoff=4, maxIts=250, keepRatio=0.10):
    origSize = len(ptsX)
    oldSize = len(ptsX) + 1
    groupSize = oldSize -1
    Its = 1
    groupList = []
    groupList.append(oldSize)
    while Its < maxIts and oldSize > np.floor(origSize*keepRatio):
        oldSize = groupSize
        stdX = np.std(ptsX)
        stdY = np.std(ptsY)
        boundsX = (np.mean(ptsX)-stdX*stdevCutoff,np.mean(ptsX)+stdX*stdevCutoff)
        boundsY = (np.mean(ptsY)-stdY*stdevCutoff,np.mean(ptsY)+stdY*stdevCutoff)
        ptsBoth = [x for x in zip(ptsX,ptsY)]
        ptsBoth = [s for s in ptsBoth if boundsX[0] < s[0] < boundsX[1] and boundsY[0] < s[1] < boundsY[1]]
        ptsX = [s[0] for s in ptsBoth]
        ptsY = [s[1] for s in ptsBoth]
        groupSize = len(ptsX)
        groupList.append(groupSize)
        Its = Its + 1
    return ptsX,ptsY,groupList



#Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in range(3))
rawImageList=Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in stackIDList)

detectDSdimensions = (1000,1000)
drawMatchNum = 500
imA = 0
imB = 3

orbTesta = featExtractORB(rawImageList[imA].resize(detectDSdimensions,resample=Image.BILINEAR))
orbTestb = featExtractORB(rawImageList[imB].resize(detectDSdimensions,resample=Image.BILINEAR))

bfNorm = cv2.BFMatcher(cv2.NORM_L2) # ignore this one. it's for SIFT / SURF
bfHamm = cv2.BFMatcher(cv2.NORM_HAMMING)

#If this is insufficient, use the knnMatch method and add 'k=2' as a parameter
matchesORB = bfHamm.match(orbTesta[1],orbTestb[1])

goodORB = []
for m in matchesORB:
    if m.distance < np.mean([s.distance for s in matchesORB])-2*np.std([s.distance for s in matchesORB]):
        goodORB.append(m)

outImga=np.array(rawImageList[imA].resize(detectDSdimensions,resample=Image.BILINEAR))
outImgb=np.array(rawImageList[imB].resize(detectDSdimensions,resample=Image.BILINEAR))

imCompORB = cv2.drawMatches(outImga,orbTesta[0],outImgb,orbTestb[0], goodORB[:], None, flags=2)

shimage(imCompORB)

#Navigating the dmatch structure
distances=[goodORB[s].distance for s in range(len(goodORB))]

idxs=[(goodORB[s].queryIdx,goodORB[s].trainIdx) for s in range(len(goodORB))]

list_ptA = [orbTesta[0][s.queryIdx].pt for s in goodORB]
list_ptB = [orbTestb[0][s.trainIdx].pt for s in goodORB]

diffsX = [list_ptB[s][0]-list_ptA[s][0] for s in range(len(list_ptA))]
diffsY = [list_ptB[s][1]-list_ptA[s][1] for s in range(len(list_ptA))]


trimX,trimY,trimList =removeOutliers2D(diffsX,diffsY,stdevCutoff=2)

hmap,xedge,yedge = np.histogram2d(trimX,trimY, bins=20)
extents=[xedge[0],xedge[-1],yedge[0],yedge[-1]]
plt.imshow(hmap.T,extent=extents,origin='lower',cmap='jet')
figORB2=plt.scatter(trimX,trimY,c=distances,cmap='hsv')

