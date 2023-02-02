#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2 
import sys 

#!pip install opencv-contrib-python --user


# In[7]:


class BackgroundSubstractionAlgo:
    framePrv=[] #Previous Frame
    frameRef=[] #Image Reference  
    def __init__(self):
        self.frames=[]
        self.isFramesFull=False
        
       #-------------------Create model GMM and MOG and MOG2-----------------------------------
        self.mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, True)
        #self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.gmmSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(10, .5)
        self.mogSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(300)
        
        #self.framePrv=[]
        #self.framePrv=[]
    
    def modify_frame(self,frame,threshold):
        modifyFrame =  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        th , modifyFrame = cv2.threshold(modifyFrame, threshold, 255, cv2.THRESH_BINARY)
        modifyFrame=cv2.resize(modifyFrame, (640,480), interpolation=cv2.INTER_AREA)
        return modifyFrame
    
    
    #-------------------------------------------------------------------------------    
    def frame_differencing(self , frame ,threshold):
        if not self.isFramesFull:
            self.frameRef=frame #Image Reference  
            self.isFramesFull=True
            
        mask_foreground=cv2.absdiff(self.frameRef,frame)
        #ret,thresh_foreground = cv2.threshold(mask_foreground,60,255,cv2.THRESH_BINARY)
        mask_foreground_binary=self.modify_frame(mask_foreground,threshold)
        cv2.imshow('FRAME DIFFERENCING',mask_foreground_binary)
        
        
    #--------------------------------------------------------------------------------   
    def temporal_derivation(self, frame,threshold):
        
        #print(frame)
        #cv2.imshow("Temporal Derivation",frame)
        if(not self.isFramesFull):
            self.framePrv=frame  #Previous Frame
            print(self.isFramesFull)
           # print(self.framePrv)
            print("-------------------------------------------")
            self.isFramesFull=True
        else:
            mask_foreground=cv2.absdiff(self.framePrv,frame)
            #ret,thresh_foreground = cv2.threshold(mask_foreground,60,255,cv2.THRESH_BINARY)
            mask_foreground_binary=self.modify_frame(mask_foreground,threshold)
            cv2.imshow('TEMPORAL DERIVATION',mask_foreground_binary)
            self.framePrv=frame
            
    #---------------------------------------------------------------------------
    def summ(self,n):
        s=np.zeros((self.frames[0].shape[0],self.frames[0].shape[1],self.frames[0].shape[2]))
        for i in range(n):
            s+=self.frames[i]
        return s
    
    def mean(self,n):
        return 1/n*self.summ( n )
    
    def moving_average(self,frame,threshold,numFrame):
        if not self.isFramesFull:
            if len(self.frames) < numFrame :
                self.frames.append(frame)
                #print(len(self.frames))
                meanFrame=self.mean(len(self.frames) ).astype('uint8')
                mask_foreground=cv2.absdiff(meanFrame,frame)
                #print(meanFrame)
                #foreground= modify_frame(frame,threshold)
                
        if len(self.frames) == numFrame :
            meanFrame=self.mean(len(self.frames) ).astype('uint8')
            mask_foreground=cv2.absdiff(meanFrame,frame).astype(np.uint8)
            self.frames.pop(0)
            self.frames.append(frame)
            self.isFramesFull=True
        
        mask_foreground_binary=self.modify_frame(mask_foreground,threshold)
        cv2.imshow('MOVING AVERAGE',mask_foreground_binary)
        
    #-----------------------------------------------------------------
  
    def filter_median(self,frame,threshold,numFrame):
        if not self.isFramesFull:
            self.frames.append(frame)
            median_model = np.median(self.frames, axis=0).astype(dtype=np.uint8)
            mask_foreground=cv2.absdiff(median_model,frame)
            
            
        if len(self.frames) == numFrame :
            median_model = np.median(self.frames, axis=0).astype(dtype=np.uint8)
            mask_foreground=cv2.absdiff(median_model,frame)
            self.frames.pop(0)
            self.frames.append(frame)
            self.isFramesFull=True
            
        mask_foreground_binary=self.modify_frame(mask_foreground,threshold)
        cv2.imshow('FILTER MEDIAN',mask_foreground_binary)
    #------------------------------------------------------------------------------------------
    
    def running_average(self,frame,threshold,numFrame ,alpha=.3 ):
        if not self.isFramesFull:
            self.frames.append(frame)
            meanFrame=self.mean(len(self.frames) ).astype('uint8')
            #print(len(self.frames))
            merge_two_frame = cv2.addWeighted(meanFrame,(1-alpha),frame,alpha,0)
            mask_foreground=cv2.absdiff(merge_two_frame,frame)
            
        if len(self.frames) == numFrame :
            meanFrame=self.mean( len(self.frames) ).astype('uint8')
            merge_two_frame = cv2.addWeighted(meanFrame,(1-alpha),frame,alpha,0)
            mask_foreground=cv2.absdiff(merge_two_frame,frame)
            self.frames.pop(0)
            self.frames.append(frame)
            self.isFramesFull=True
                #print(meanFrame)
                #foreground= modify_frame(frame,threshold)
                
        mask_foreground_binary=self.modify_frame(mask_foreground,threshold)
        cv2.imshow('RUNNING AVERAGE',mask_foreground_binary)

#-----------------------------------------------MOG2------------------------------------------------------------------------
    def method_MOG2(self,frame):
        
        resizeFrame=cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)
        mog2Mask = self.mog2Subtractor.apply(resizeFrame)
        cv2.imshow('Background Subtraction MOG2',mog2Mask)

#-----------------------------------------------MOG------------------------------------------------------------------------
    def method_MOG(self,frame):
        
        resizeFrame=cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)
        mogMask = self.mogSubtractor.apply(resizeFrame)
        cv2.imshow('Background Subtraction MOG',mogMask)
        
#-----------------------------------------------MOG2------------------------------------------------------------------------
    def method_GMM(self,frame):
        
        resizeFrame=cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)
        gmmMask = self.gmmSubtractor.apply(resizeFrame)
        gmmMask = cv2.morphologyEx(gmmMask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        cv2.imshow('Background Subtraction GMM',gmmMask)
        
 
         
#------------------------------------------------------------------------------------------------------------------------

def choiseOperation(algorithme,operation,threshold,numFrame,currentFrame) :  #,imgBackground
    
    if(operation =="Frame_differencing"):
        algorithme.frame_differencing(currentFrame , threshold )
    elif(operation=="Temporal_derivation"):
        algorithme.temporal_derivation(currentFrame,threshold)
    elif(operation=="Moving_average"):
        algorithme.moving_average(currentFrame,threshold,numFrame)
    elif(operation=="Filter_median"):
        algorithme.filter_median(currentFrame,threshold,numFrame)
    elif(operation=="Running_average"):
        algorithme.running_average(currentFrame,threshold,numFrame )
    elif(operation=="MOG_method"):
        algorithme.method_MOG(currentFrame)
    elif(operation=="MOG2_method"):
        algorithme.method_MOG2(currentFrame)
    elif(operation=="GMM_method"):
        algorithme.method_GMM(currentFrame)
        
    else:
        print("Operation not supported !")
        


# In[8]:


class ReadVideo:
    def __init__(self,path=None,isFromWebCam=False):
        self.path=path
        self.isFromWebCam=isFromWebCam
        self.cap=""
        self.threshold=20
        self.numFrame=15
        self.imgBackground=""
        self.algorithme=BackgroundSubstractionAlgo()
        
    
    def load(self):
        if not self.isFromWebCam:
            self.cap = cv2.VideoCapture(self.path)
        else:
            self.cap = cv2.VideoCapture(0)
            
            
    def capture(self,operation):
        if(self.cap.isOpened() == False):
            print("Error opening video stream or file")
            sys.exit(0)
        while(self.cap.isOpened()):
            ret, frame_current = self.cap.read()
            
            if ret :
                #currentFrame = cv2.resize(currentFrame,(600,450))
                print(operation)
                choiseOperation(self.algorithme,operation,self.threshold,self.numFrame,frame_current)
                cv2.namedWindow("Main")
                modifyFrame=cv2.resize(frame_current, (640,480), interpolation=cv2.INTER_AREA)
                cv2.imshow("Main", modifyFrame)
                if cv2.waitKey(25) & 0XFF ==  ord('q'):
                    break
            else:
                print('no video')
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.release()
        self.cleanup()
        
    def setThreshold(self,thresh):
        self.threshold=thresh
        
    def setNbrOfFrame(self,nbrFrame):
        self.numFrame=nbrFrame
        
    def setImgBackground(self,imgBackground):
        self.imgBackground=imgBackground
        
    def setIsFromWebCam(self,boolean):
        self.isFromWebCam=boolean
        
    def setPath(self,path):
        self.path=path
        
    def release(self):
        self.cap.release()

    def cleanup(self):
        cv2.destroyAllWindows()

