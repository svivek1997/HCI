# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:13:21 2019

@author: Tarun
"""

import cv2
import numpy as np

green_lowerbound=np.array([50,120,130])
green_upperbound=np.array([102,255,255])
kernel = np.ones((5,5))
kernelclose = np.ones((10,10))
cap = cv2.VideoCapture(0)
img = np.zeros( (480, 640, 3),np.uint8 )
img[:]=(255,255,255)
flag=0;

while(cap.isOpened()):
    ret,frame = cap.read();
    #print(frame.shape)
    frame = cv2.flip(frame,1)
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #create the mask
    mask = cv2.inRange(frame_hsv,green_lowerbound,green_upperbound)
    
    #cleaning noise
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel);
    maskClose = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernel);
    #cv2.imshow("img",img)
    
    maskfinal = maskClose
    _,conts,h = cv2.findContours(maskfinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    if len(conts)>0:
        cnt = sorted(conts, key = cv2.contourArea, reverse = True)[0]
        #for i in range(len(conts)):
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if flag==0:
            cv2.circle(img,( int((x+x+w)/2),int((y+y+h)/2) ) ,2,(0,0,0),-1)
            flag=1
            gpoint=( int((x+x+w)/2),int((y+y+h)/2) )
        else:
            cv2.line(img,( int((x+x+w)/2),int((y+y+h)/2)) ,gpoint,(0,0,0),2,-1)
            gpoint=( int((x+x+w)/2),int((y+y+h)/2) )
    else:
        flag=0
        
    
    #mask2= cv2.inRange(frame,np.array([50,50,50]),np.array([20,255,20]))
    #cv2.imshow("closed",maskClose)
    cv2.imshow("img",img)
    #cv2.imshow("maskopen",maskOpen)
    cv2.imshow("bgr",frame)
    if cv2.waitKey(1)==27:
        break;
        
cv2.destroyAllWindows()
cap.release()