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
featStack = list(range(sliceNum))

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

#need to rejigger this to make it able to handle an input of the matches that
    #survive the initial distance cutoff in the previous processing step.
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

def registerImages(imA,imB,method='ORB', DSdims=(1000,1000),maxMatchNum=500):
    detectDSdimensions = DSdims
    drawMatchNum = maxMatchNum
    imA = 0
    imB = 1
    orbTesta = featExtractORB(rawImageList[imA].resize(detectDSdimensions,resample=Image.BILINEAR))
    orbTestb = featExtractORB(rawImageList[imB].resize(detectDSdimensions,resample=Image.BILINEAR))
    bfNorm = cv2.BFMatcher(cv2.NORM_L2) # ignore this one. it's for SIFT / SURF
    bfHamm = cv2.BFMatcher(cv2.NORM_HAMMING)
    matchesORB = bfHamm.match(orbTesta[1],orbTestb[1])
    
    
    
    #Here is the spot for putting in the coordinates for everything.
    #The structure should be something like matches, ptsa, ptsb, goodMatch, useForTxBool
    #They can remain as different structures as long as we keep everything to a list
    # that is the length of the matches. That might need to be trimmed later, but it's
    # better to have it and not need it than the converse.
    
    goodORB = []
    for m in matchesORB:
        if m.distance < np.mean([s.distance for s in matchesORB])-2*np.std([s.distance for s in matchesORB]):
            goodORB.append(m)
#    for m in len(matchesORB):
#        if goodORB[m].distance < np.mean([s.distance for s in matchesORB]):
#            



    outImga=np.array(rawImageList[imA].resize(detectDSdimensions,resample=Image.BILINEAR))
    outImgb=np.array(rawImageList[imB].resize(detectDSdimensions,resample=Image.BILINEAR))
    qcFig,((qcAx1,qcAx2),(qcAx3,qcAx4)) = plt.subplots(nrows=2,ncols=2)

    imCompORB = cv2.drawMatches(outImga,orbTesta[0],outImgb,orbTestb[0], goodORB[:], None, flags=2)
    
    #shimage(imCompORB)
    qcAx1.imshow(imCompORB)
    
    #Navigating the dmatch structures
    distances=[goodORB[s].distance for s in range(len(goodORB))]
    
    idxs=[(goodORB[s].queryIdx,goodORB[s].trainIdx) for s in range(len(goodORB))]
    
    list_ptA = [orbTesta[0][s.queryIdx].pt for s in goodORB]
    list_ptB = [orbTestb[0][s.trainIdx].pt for s in goodORB]
    
    diffsX = [list_ptB[s][0]-list_ptA[s][0] for s in range(len(list_ptA))]
    diffsY = [list_ptB[s][1]-list_ptA[s][1] for s in range(len(list_ptA))]
    
    
    trimX,trimY,trimList =removeOutliers2D(diffsX,diffsY,stdevCutoff=2)
    
    hmap,xedge,yedge = np.histogram2d(trimX,trimY, bins=20)
    extents=[xedge[0],xedge[-1],yedge[0],yedge[-1]]
    qcAx3.imshow(hmap.T,extent=extents,origin='lower',cmap='jet')
    qcAx3=plt.scatter(trimX,trimY) #need the indices here



#The actual code to be run.
    
#Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in range(3))
rawImageList=Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in stackIDList)

#Get the features into a stack (kps, desc, imageOfKeypoints) x sliceNum
detectDSdimensions=(1000,1000)
#featStack=Parallel(n_jobs=12)(delayed(featExtractORB)(ID) for ID in rawImageList)
for featSlice in range(len(featStack)):
    featStack[featSlice]=featExtractORB(rawImageList[featSlice].resize(detectDSdimensions,resample=Image.BILINEAR))

#making the matching loop that goes through and gets the matches across all the combinations
    #For a given span (currently hardcoded at 3 before and after)
bfHamm = cv2.BFMatcher(cv2.NORM_HAMMING)
matchStack=list(range(sliceNum))

regSpan = 5

matchNumStack=np.zeros((sliceNum,regSpan*2+1))

#WORK ON PARALLELIZING THIS
#write a function that will take in a slice and the span and find the matches of the surrounding ones and 
#spit out a row that has all the match aspects (probably trim it in here too.)
for curSlice in range(sliceNum):
    curMatchList=[]
    for adjSlice in [s for s in list(range(curSlice-regSpan,curSlice+1+regSpan)) if s!=curSlice]:
        if 0<= adjSlice <sliceNum:
            print(curSlice, adjSlice)
            curMatch=bfHamm.match(featStack[curSlice][1],featStack[adjSlice][1])
            goodMatch = []
            for m in curMatch:
                if m.distance < np.mean([s.distance for s in curMatch])-2*np.std([s.distance for s in curMatch]):
                    goodMatch.append(m)
            curMatchList.append(goodMatch)
            matchNumStack[curSlice,adjSlice-curSlice+regSpan]=len(goodMatch)
    matchStack[curSlice]=curMatchList            

#Function of the above. Returns an image of the matches, image of the displacements, 
# the actual diplacements, and the geometric transform between each pair.
# Should be sufficient.
    #For input shortcuts, I'm making it so that you just put in the two entries from
    #the featStack object, which contains the keypoints, the descriptors, and the
    #QC image made by 'drawkeypoints' function of cv2.
def registerImPair(inputA,inputB,showMatches):
    (kpsA,descA,imA)=inputA
    (kpsB,descB,imB)=inputB
    curMatch=bfHamm.match(descA,descB)
    goodMatch=[]
    for m in curMatch:
        if m.distance < np.mean([s.distance for s in curMatch])-2*np.std([s.distance for s in curMatch]):
                    goodMatch.append(m)
    #qcFig,((qcAx1,qcAx2),(qcAx3,qcAx4)) = plt.subplots(nrows=2,ncols=2)
    
    #Need to remove the outliers without losing my correct indices. Or else go ahead and calculate the 
    #geo transform before anything else.
    
    
    
    if showMatches:
        outImga=inputA[2]
        outImgb=inputB[2]
        imCompORB = cv2.drawMatches(outImga,inputA[0],outImgb,inputB[0], goodMatch[:], None, flags=2)
        
        shimage(imCompORB)
    
    return goodMatch

#

#Can't parallelize because pickle doesn't work on a lot of types of stuff. So we're going to have to 
    # complete the estimation of the transform and then save the transforms. Probably as matrices
    # because pickles.
def registerNeighborhood(curSlice,regSpan,QCBool):
    neighborhood = list(range(regSpan*2+1))
    for adjSlice in [s for s in list(range(curSlice-regSpan,curSlice+1+regSpan)) if s!=curSlice]:
        if 0<=adjSlice<sliceNum:
            curReg=registerImPair(featStack[curSlice],featStack[adjSlice])
            neighborhood[adjSlice-curSlice+regSpan]=curReg
            if QCBool:
                print("images everywherrrre")
        else:
            neighborhood[adjSlice-curSlice+regSpan]=[]
    return neighborhood

test=Parallel(n_jobs=12)(delayed(registerNeighborhood)(ID,5,0) for ID in stackIDList)
