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
    

#Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in range(3))
rawImageList=Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in stackIDList)

detectDSdimensions = (1000,1000)
drawMatchNum = 500
imA = 0
imB = 3

akazeTesta = featExtractAKAZE(rawImageList[imA].resize(detectDSdimensions,resample=Image.BILINEAR))
akazeTestb = featExtractAKAZE(rawImageList[imB].resize(detectDSdimensions,resample=Image.BILINEAR))

orbTesta = featExtractORB(rawImageList[imA].resize(detectDSdimensions,resample=Image.BILINEAR))
orbTestb = featExtractORB(rawImageList[imB].resize(detectDSdimensions,resample=Image.BILINEAR))

bfNorm = cv2.BFMatcher(cv2.NORM_L2)
bfHamm = cv2.BFMatcher(cv2.NORM_HAMMING)

matchesORB = bfHamm.knnMatch(orbTesta[1],orbTestb[1], k=2)
matchesAKAZE = bfNorm.knnMatch(akazeTesta[1],akazeTestb[1], k=2)

goodAKAZE = []
for m,n in matchesAKAZE:
    if m.distance < 0.9*n.distance:
        goodAKAZE.append([m])

goodORB = []
for m,n in matchesORB:
    if m.distance < 0.9*n.distance:
        goodORB.append([m])

outImga=np.array(rawImageList[imA].resize(detectDSdimensions,resample=Image.BILINEAR))
outImgb=np.array(rawImageList[imB].resize(detectDSdimensions,resample=Image.BILINEAR))

imCompAKAZE = cv2.drawMatchesKnn(outImga,akazeTesta[0],outImgb,akazeTestb[0],
                         goodAKAZE[1:drawMatchNum], None, flags=2)

imCompORB = cv2.drawMatchesKnn(outImga,orbTesta[0],outImgb,orbTestb[0],
                         goodORB[1:drawMatchNum], None, flags=2)

shimage(imCompAKAZE)
shimage(imCompORB)