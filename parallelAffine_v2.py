
import numpy as np
import sys
import cv2
import os
from PIL import Image
from PIL import ImageOps
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
#from scipy.misc import imread
### Parameters
if sys.argv[0]:
    system=sys.argv[0]
debug=0

#These will change each time that the script is used
if system=='win': 
    srcDir = "C:\\Users\\karlf\\Documents\\Data\\ixQ\\MasterRaw\\ixQ\\waf010_BSD_64nm"
    dstDir = "C:\\Users\\karlf\\Documents\\Data\\pyAlignEM\\working"
if system=='lin':
    srcDir = '/media/karl/OS/Users/karlf/Documents/Data/ixQ/MasterRaw/ixQ/waf010_BSD_64nm'
    srcDir = '/media/karl/OS/Users/karlf/Documents/Data/ixQ/MasterRaw/ixQ/ixQ_waf009_IL_04nm_deepIPL_01'
    dstDir = '/media/karl/OS/Users/karlf/Documents/Data/pyAlignEM/working'

sliceNum=48

#These will change slightly less often
DSsize=(8192,8192)
regSpan = 5
boxDim=2048
#cropBox=(4096-boxDim,4096,4096,4096+boxDim)
cropBox=(0,0,8192,8192)

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
def shimage(image,resizeDims=(1024,1024)):
    if isinstance(image.size,int):
        img = Image.fromarray(image)
    img.resize(resizeDims)
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
def removeOutliers2D(matchDat, stdevCutoff=2, maxIts=250, keepRatio=0.01, absMin=10):
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
        boundsX = (np.median(xdiffs)-stdX*stdevCutoff,np.median(xdiffs)+stdX*stdevCutoff)
        boundsY = (np.median(ydiffs)-stdY*stdevCutoff,np.median(ydiffs)+stdY*stdevCutoff)
        if len(keepDat) > np.floor(origSize*keepRatio) and len(keepDat) > absMin:
            keepDat = [s for s in keepDat \
                       if boundsX[0] < s[4] < boundsX[1] \
                       and boundsY[0] < s[5] < boundsY[1]] 
        groupSize = len(xdiffs)
        groupList.append(groupSize)
        Its = Its + 1
        
        #add a plot function and call it here to watch the decrease in the lsit of accepted pts.
        plotMatches(keepDat)
        
        #plt.scatter([s[4] for s in keepDat],[s[5] for s in keepDat])
    return keepDat

def removeImproved(matchDat, matchMin=50, absMin=50, margin=200, stdevs=2):
    #Add code for adaptive neighborhood density. Start with unattainable and relax until group is found.
    #This will allow to get the nearest group. Unless large rotation, this should work fine.
    #Need to deal with large rotation if necessary
    keepDat=matchDat
    keepBin=[0]*len(keepDat)
    xdiffs=[s[4] for s in keepDat]
    ydiffs=[s[5] for s in keepDat]
    counts,xedge,yedge=np.histogram2d(xdiffs,ydiffs,bins=100)
    result = np.where(counts == np.amax(counts))
    listOfCordinates = list(zip(result[0], result[1]))
    xbin=xedge[listOfCordinates[0][0]]
    ybin=yedge[listOfCordinates[0][1]]
    for mp in range(len(keepDat)):
        cxd = keepDat[mp][4]
        cyd = keepDat[mp][5]
        if xbin-(margin*0.5) < cxd < xbin+(margin*1.5) and ybin-(margin*0.5) < cyd < ybin+(margin*1.5):
            keepBin[mp]=1
    stdevx = np.std([s for s,x in zip(xdiffs,keepBin) if x])
    stdevy = np.std([s for s,x in zip(ydiffs,keepBin) if x])
    stmeanx = np.mean([s for s,x in zip(xdiffs,keepBin) if x])
    stmeany = np.mean([s for s,x in zip(ydiffs,keepBin) if x])
    
    for mp in range(len(keepDat)):
        cxd = keepDat[mp][4]
        cyd = keepDat[mp][5]
        if stmeanx-(stdevx*stdevs)<cxd<stmeanx+(stdevx*stdevs) \
         and stmeany-(stdevy*stdevs)<cyd<stmeany+(stdevy*stdevs):
            keepBin[mp]=1
        else:
            keepBin[mp]=0
    print(np.sum(keepBin))
    results=[i for (i,v) in zip(matchDat,keepBin) if v]
    if debug==1:
        plotMatches(results)
    return results

def registerSubStack(curSlice, minRegPts=5):
    #get a list of the adjacent slices needing registration
    subStackTFs=[np.array([[0,0,0],[0,0,0],[0,0,0]])]*(regSpan*2+1)
    adjSliceList=[s for s in list(range(curSlice-regSpan,curSlice+1+regSpan)) if s!=curSlice]
    #initiate the matcher
    locHamm=cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=True)
    #Go through and do the matching for each of the relevant pairs
    # This could be better if I didn't do the redundant back-matching.
    for adjSlice in adjSliceList:
        print(adjSlice)
        if 0<=adjSlice<sliceNum: #This controls whice adjSlices are reg'd
            #create the matches
            curMatch=locHamm.match(featStack[curSlice][1],featStack[adjSlice][1])
            #get a pared down list ready
            goodMatch=[]
            #This loop was giving too few matches that weren't a lot better than just taking the sorted list
#            for m in curMatch:
#                if m.distance < np.mean([s.distance for s in curMatch])-2*np.std([s.distance for s in curMatch]):
#                    goodMatch.append(m)    #If I don't like the results from the cutoff, I can just sort and thresh
            goodMatch = sorted(curMatch, key = lambda x:x.distance)[:500]
            #Build the data structure that has the IDs, the euclidean pts, and the translation difference
            matchDat=[(goodMatch[s].queryIdx,goodMatch[s].trainIdx, \
             featStack[curSlice][0][goodMatch[s].queryIdx][2],featStack[adjSlice][0][goodMatch[s].trainIdx][2], \
             featStack[curSlice][0][goodMatch[s].queryIdx][2][0]-featStack[adjSlice][0][goodMatch[s].trainIdx][2][0], \
             featStack[curSlice][0][goodMatch[s].queryIdx][2][1]-featStack[adjSlice][0][goodMatch[s].trainIdx][2][1]) \
             for s in range(len(goodMatch))]
            if debug==1:
                plotMatches(matchDat)
            #Use the histogram of the displacements to find a concentration of matches that all indicate the same
            #general transform (at least for translation)
            #This will BREAK if there is a significant rotation to the Tx
            consensusMatchDat=removeImproved(matchDat,stdevs=2,margin=200)
            if debug==1:
                plotMatches(consensusMatchDat)
            #Only attempt to get the TF if there are sufficient pts to do it. Defaults to 5.
            if len(consensusMatchDat)>minRegPts:
                curTF,status=cv2.findHomography(np.array([s[2] for s in consensusMatchDat]),np.array([s[3] for s in consensusMatchDat]))
                subStackTFs[adjSlice-curSlice+regSpan]=curTF
    return subStackTFs

def parFeat(sliceID):
    featSliceImage = rawImageList[sliceID]
    featSliceImage = featSliceImage.crop(cropBox)
    [kps,fts,desc]=featExtractORB(featSliceImage)
    kpsTups=[(s.angle,s.octave,s.pt,s.response,s.size) for s in kps]
    feats=[kpsTups,fts,desc]
    print(sliceID)
    return feats

def plotMatches(consensusMatchDat):
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(212)
    ax1.hist2d([p[4] for p in consensusMatchDat],[p[5] for p in consensusMatchDat],bins=100)
    ax1.scatter([p[4] for p in consensusMatchDat],[p[5] for p in consensusMatchDat])
    fig2 = plt.figure(figsize=(5,1))
    ax2 = fig2.add_subplot(111)
    ax2.hist([p[4] for p in consensusMatchDat],bins=100)
    ax2.hist([p[5] for p in consensusMatchDat],bins=100)
    return fig,fig2

def getGlobalTFs(anchorSlice=np.floor(sliceNum/2)):
    #make the array to store the final transforms
    finalTFs=[np.array([[0,0,0],[0,0,0],[0,0,0]])]*(sliceNum)
    finalTFs[anchorSlice]=np.float64(np.array([[1,0,0],[0,1,0],[0,0,1]]))
    #the loop going backward from anchorSlice
    for curSlice in reversed(stackIDList[:anchorSlice]):
        print(curSlice)
        if np.abs(allTFs[curSlice][6][0][0]+allTFs[curSlice][6][0][0]-2)<0.25:
            finalTFs[curSlice]=np.matmul(allTFs[curSlice][6],finalTFs[curSlice+1])
            
    #the loop going forward from anchorSlice
    for curSlice in stackIDList[anchorSlice+1:]:
        print(curSlice)
        if np.abs(allTFs[curSlice][4][0][0]+allTFs[curSlice][4][0][0]-2)<0.25:
            finalTFs[curSlice]=np.matmul(allTFs[curSlice][4],finalTFs[curSlice-1])
    return finalTFs

def showRegistered():
    blank=rawImageList[0]
    blankra=np.asarray(blank)*0
    cv2.imshow('reggd',blankra)
    for curSlice in stackIDList:
        tf=finalTFs[curSlice]
        cvim=rawImageList[curSlice].crop(cropBox)
        cvimra=np.asarray(cvim)
        imwrpra=cv2.warpPerspective(cvimra,tf,(8192,8192))
        imwrprasm=cv2.resize(imwrpra,(1024,1024))
        cv2.imshow('reggd', imwrprasm)
        print(curSlice)
        time.sleep(1)
        cv2.waitKey(1)
        
    return 0

def showRegNumpy():
    for curSlice in range(len(stackIDList)):
        tf=finalTFs[curSlice]
        img=rawImageList[curSlice]
        imgl=img.convert("L")
        imglra=np.asarray(imgl)
        imwrp=cv2.warpPerspective(imglra,tf,(8192,8192))
        imwrp=Image.fromarray(imwrp)
        #img=imwrp.resize((1024,1024))
        img=imwrp.crop((3584,3584,4608,4608))
        plt.figure(1); plt.clf()
        plt.imshow(np.asarray(img), cmap='gray')
        plt.title(curSlice)
        plt.pause(1)
    return 0
        
##################################
### The actual code for running things
### I should really put this into a main()
##################################

#Load and downsample the raw images
rawImageList=Parallel(n_jobs=12)(delayed(loadImage)(ID) for ID in stackIDList)

#Feature detection is really quick, and the parallelization makes it slower.

#The qc stack exists solely to preserve the kps format for doing match drawing
#DELETE THIS LATER
qcStack=featStack[:]

serialFeat=0
if serialFeat:
    #I wrapped this loop into a function for parallelization
    for featSlice in range(len(featStack)):
        #get the image from the import stack
        featSliceImage = rawImageList[featSlice]
        #crop / resize / whatever
        featSliceImage = featSliceImage.crop(cropBox)
        #extract features
        [kps,fts,desc]=featExtractORB(featSliceImage)
        #get the format picklable for parallelization
        kpsTups=[(s.angle,s.octave,s.pt,s.response,s.size) for s in kps]
        #put it in the big feature stack
        featStack[featSlice]=[kpsTups,fts,desc]
        #Also delete this part
        qcStack[featSlice]=[kps,fts,desc]
        print(featSlice)
else:
    #this is the parallelized version of the above loop
    featStack=Parallel(n_jobs=12)(delayed(parFeat)(ID) for ID in stackIDList)



#This is getting all the affine transforms for all combinations of neighborhoods.
serialReg=0
#can't do parallel with windows
if system=='win':
    serialReg=1

if serialReg:
    allTFs=[]
    for curSlice in stackIDList:
        curTFs=registerSubStack(curSlice)
        allTFs.append(curTFs)
        print(curSlice)
else:
    #This is crashing with some bizarre win32 file in use error
    allTFs=Parallel(n_jobs=12)(delayed(registerSubStack)(ID) for ID in stackIDList)

#Get the right output filname
if system=='win':
    outputFile=dstDir+"\\allTFs_parallel3"
if system=='lin':
    outputFile=dstDir+'/allTFs_parallel3'

#save the TFs incase they are needed later.
np.save(outputFile,allTFs)
    
anchorSlice=24

finalTFs=getGlobalTFs(anchorSlice)

#showRegNumpy()



#Test code for wrapping head around transforms
def cvConvert(inputImage):
    imarray=np.array(inputImage,dtype='uint8')
    imarrayRGB=np.array([imarray,imarray,imarray])
    imarrayRGBt=np.transpose(imarrayRGB,(1,2,0))
    cvImage=cv2.cvtColor(imarrayRGBt,cv2.COLOR_BGR2GRAY)
    return cvImage

def registerImPair(inputA,inputB,showMatches):
    (kpsA,descA,imA)=inputA
    (kpsB,descB,imB)=inputB
    bfHamm = cv2.BFMatcher(cv2.NORM_HAMMING)
    curMatch=bfHamm.match(descA,descB)
    goodMatches=[]
    for m in curMatch:
        if m.distance < np.mean([s.distance for s in curMatch])-2*np.std([s.distance for s in curMatch]):
                    goodMatches.append(m)
    print(len(goodMatches))
    if showMatches:
        outImga=inputA[2]
        outImgb=inputB[2]
        imCompORB = cv2.drawMatches(outImga,inputA[0],outImgb,inputB[0], goodMatches[:], None, flags=2)
        
        shimage(imCompORB,(2048,1024))
    
    return goodMatches
#
#tfScaleList=[]
#for cs in stackIDList:
#    csTF=allTFs[cs]
#    tfScaleRow=[]
#    ct = csTF[6]
#    if np.size(ct)>1:
#        tfScaleRow.append([ct[0][0],ct[1][1]])
#    else:
#        tfScaleRow.append(0)
#    tfScaleList.append(tfScaleRow)
#
#cvIm3=cvConvert(rawImageList[5].crop(cropBox))
#cvIm4=cvConvert(rawImageList[8].crop(cropBox))
#
#tf=allTFs[5][8]
#
#cvIm3wrp=cv2.warpPerspective(cvIm3,tf,(8192,8192))
#
#cvImDiff=cv2.merge((cvIm3wrp,cvIm4,cvIm3*0))
#
#cvImSBS=np.concatenate((cvIm3wrp,cvIm4),axis=1)
#
#shimage(cvImDiff)
#
#shimage(cv2.resize(cvImSBS,(2048,1024)))
#
#ignore=registerImPair(qcStack[9],qcStack[4],1)
#
#anchorSlice=10
##Now that all the transforms are available, need to go through and get them set
##Could try a wobbly mesh / force directed thing. Or just take the average of the transforms to the next five stable ones.
#finalTF=[None]*sliceNum
#finalTF[anchorSlice]=np.float64([[1,0,0],[0,1,0],[0,0,1]])
#pathList=[[-4,5,0,-4],[-3,4,0,-3],[-2,3,0,-2],[-1,2,0,-1],[0,1,0,0],[1,0,0,1],[2,-1,0,2],[3,-2,0,3],[4,-3,0,4],[5,-4,0,5]]
#pathList=[[0,1,0,0],[1,0,0,1],[2,-1,0,2],[3,-2,0,3],[4,-3,0,4],[5,-4,0,5]]
#for curSlice in reversed(range(anchorSlice)):
##    TF1=allTFs[curSlice][regSpan+1] #goto slice+1
##    TF2=np.matmul(allTFs[curSlice+2][regSpan-1],allTFs[curSlice][regSpan+2]) #Go to slice+2 then back 1
##    TF3=np.matmul(allTFs[curSlice+3][regSpan-2],allTFs[curSlice][regSpan+3]) #Go to slice+3 then back 2
#
#    compositeTFs=[None]*(regSpan*2+1)
#    for none in range(len(compositeTFs)):
#        compositeTFs[none] = 0
##    adjSliceList=[s for s in list(range(curSlice-regSpan,curSlice+1+regSpan)) if s!=curSlice]
##    for adjSlice in adjSliceList:
##        print(adjSlice)
##        if 0<=adjSlice<sliceNum:
#    for path in range(len(pathList)):
#        pathLocs=pathList[path]
#        if np.size(allTFs[curSlice+pathLocs[0]][regSpan+pathLocs[1]])>1 and np.size(allTFs[curSlice+pathLocs[2]][regSpan+pathLocs[3]])>1:
#            pathTF=np.matmul(allTFs[curSlice+pathLocs[0]][regSpan+pathLocs[1]],allTFs[curSlice+pathLocs[2]][regSpan+pathLocs[3]])
#            compositeTFs[path]=pathTF
#    finalTF[curSlice]=compositeTFs
#
