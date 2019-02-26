# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:20:24 2019

@author: karlf
"""
import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageOps
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

### Parameters

#These will change each time that the script is used
srcDir = "C:\\Users\\karlf\\Documents\\Data\\ixQ\\MasterRaw\\ixQ\\waf010_BSD_64nm"
dstDir = "C:\\Users\\karlf\\Documents\\Data\\pyAlignEM\\working"
sliceNum=24

#These will change slightly less often
DSsize=(8192,8192)
regSpan = 5
boxDim=2048
cropBox=(4096-boxDim,4096,4096,4096+boxDim)

#PIL thinks that big images are attacks
Image.MAX_IMAGE_PIXELS=1025000000
Image.MAX_IMAGE_PIXELS=None

#Setting up all the necessary lists and structures for later
dirList = os.listdir( srcDir )
dirList.sort()
sliceList = [s for s in dirList if 'waf' in s and 'Mont' in s]
stackIDList = list(range(sliceNum))
rawImageList = list(range(sliceNum))
featureList = list(range(sliceNum))
featStack = list(range(sliceNum))


### All the Functions

#This will load the mast file and downsample it to whatever size you are using as DSsize
def loadMask(maskFileFullPath):
    rawMask=Image.open(maskFileFullPath)
    if rawMask.size!=DSsize:
        rawMask.resize(DSsize,resample=Image.BILINEAR)
    return rawMask

#This loads and downsamples the raw images. The raw images are not used further.
def loadImage(stackID):
    Image.MAX_IMAGE_PIXELS=None
    curSlice=sliceList[stackID]
    fileList=os.listdir( srcDir+"/"+str(curSlice))
    fileName=[s for s in fileList if len(s)<30 and '.tif' in s][0]
    imraw=Image.open( srcDir+"/"+curSlice+"/"+fileName )
    if imraw.size[0]>=DSsize[0]:
        imDS=imraw.resize(DSsize,resample=Image.BILINEAR)
    imDSL=imDS.convert('L')
    imDSLinv=ImageOps.invert(imDSL)
    imDSinv=imDSLinv.convert('P')
    return imDSinv

#This uses the ORB feature extractor to get the features which will then be matched.
def featExtractORB(image):
    orbDetector = cv2.ORB_create()
    orbDetector.setMaxFeatures(2500)
    kps = orbDetector.detect(np.array(image), None)
    (kps, descs) = orbDetector.compute(np.array(image), kps)
    img=cv2.drawKeypoints(np.array(image), kps, None, color = (255,0,0))
    return (kps, descs, img)

#shows a numpy array as an image
def shimage(image):
    img = Image.fromarray(image)
    img.show()

#One-off for the outlier removal function.
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
def removeOutliers2D(matchDat, stdevCutoff=2, maxIts=50, keepRatio=0.20):
    keepDat=matchDat
    stdX=100
    stdY=100
    xdiffs=[s[4] for s in keepDat]
    ydiffs=[s[5] for s in keepDat]
    origSize = len(xdiffs)
    oldSize = len(xdiffs) + 1
    groupSize = oldSize -1
    Its = 1
    groupList = []
    groupList.append(oldSize)
    while Its < maxIts:
        xdiffs=[s[4] for s in keepDat]
        ydiffs=[s[5] for s in keepDat]
        oldSize = groupSize
        stdX = np.std(xdiffs)
        stdY = np.std(ydiffs)
        boundsX = (np.mean(xdiffs)-stdX*stdevCutoff,np.mean(xdiffs)+stdX*stdevCutoff)
        boundsY = (np.mean(ydiffs)-stdY*stdevCutoff,np.mean(ydiffs)+stdY*stdevCutoff)
        if len(keepDat) > np.floor(origSize*keepRatio):
            keepDat = [s for s in keepDat \
                       if boundsX[0] < s[4] < boundsX[1] \
                       and boundsY[0] < s[5] < boundsY[1]] 
        groupSize = len(xdiffs)
        groupList.append(groupSize)
        Its = Its + 1
        #plt.scatter([s[4] for s in keepDat],[s[5] for s in keepDat])
    return keepDat

def registerSubStack(curSlice):
    #get a list of the adjacent slices needing registration
    subStackTFs=[None]*(regSpan*2+1)
    adjSliceList=[s for s in list(range(curSlice-regSpan,curSlice+1+regSpan)) if s!=curSlice]
    #initiate the matcher
    locHamm=cv2.BFMatcher(cv2.NORM_HAMMING)
    #Go through and do the matching for each of the relevant pairs
    # This could be better if I didn't do the redundant back-matching.
    for adjSlice in adjSliceList:
        print(adjSlice)
        if 0<=adjSlice<sliceNum:
            #create the matches
            curMatch=locHamm.match(featStack[curSlice][1],featStack[adjSlice][1])
            #get a pared down list ready
            goodMatch=[]
            for m in curMatch:
                if m.distance < np.mean([s.distance for s in curMatch])-2*np.std([s.distance for s in curMatch]):
                    goodMatch.append(m)    #If I don't like the results from the cutoff, I can just sort and thresh
            #goodMatch = sorted(curMatch, key = lambda x:x.distance)[:200]
            
            #get the consensus matches for the current pair
            #This has the idxs and the euc pts for each of the good matches
            
            #KARL: you are looking at the wrong area, you idiot.
            matchDat=[(goodMatch[s].queryIdx,goodMatch[s].trainIdx, \
             featStack[curSlice][0][goodMatch[s].queryIdx][2],featStack[adjSlice][0][goodMatch[s].trainIdx][2], \
             featStack[curSlice][0][goodMatch[s].queryIdx][2][0]-featStack[adjSlice][0][goodMatch[s].trainIdx][2][0], \
             featStack[curSlice][0][goodMatch[s].queryIdx][2][1]-featStack[adjSlice][0][goodMatch[s].trainIdx][2][1]) \
             for s in range(len(goodMatch))]
            
            consensusMatchDat=removeOutliers2D(matchDat,stdevCutoff=3)
            curTF,status=cv2.findHomography(np.array([s[2] for s in consensusMatchDat]),np.array([s[3] for s in consensusMatchDat]))            
            subStackTFs[adjSlice-curSlice+regSpan]=curTF
            #subStackTFs[adjSlice-curSlice+regSpan]=len(goodMatch)
            
    return subStackTFs

### The actual code for things

#Load and downsample the raw images
rawImageList=Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in stackIDList)

#Feature detection is really quick, and the parallelization makes it slower.
#Detecting features serially
for featSlice in range(len(featStack)):
    #featSliceImage = rawImageList[featSlice].resize(detectDSdimensions,resample=Image.BILINEAR)
    featSliceImage = rawImageList[featSlice]
    featSliceImage = featSliceImage.crop(cropBox)
    [kps,fts,desc]=featExtractORB(featSliceImage)
    kpsTups=[(s.angle,s.octave,s.pt,s.response,s.size) for s in kps]
    featStack[featSlice]=[kpsTups,fts,desc]
    print(featSlice)

#This is crashing with some bizarre win32 file in use error. Save it for later.
#testSubPar=Parallel(n_jobs=4)(delayed(registerSubStack)(ID) for ID in stackIDList)

#This is getting all the affine transforms for everyone.
allTFs=[]
for curSlice in stackIDList:
    curTFs=registerSubStack(curSlice)
    allTFs.append(curTFs)
    print(curSlice)

anchorSlice=10
#Now that all the transforms are available, need to go through and get them set
#Could try a wobbly mesh / force directed thing. Or just take the average of the transforms to the next five stable ones.
finalTF=[None]*sliceNum
finalTF[anchorSlice]=np.float64([[1,0,0],[0,1,0],[0,0,1]])
for curSlice in reversed(range(anchorSlice)):
    TF1=allTFs[curSlice+1][regSpan-1]
    TF2=allTFs[curSlice+1][regSpan+1]*allTFs[curSlice+2][regSpan-2]    
    



