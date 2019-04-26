# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:15:59 2019

@author: Tarun
"""

from scipy.spatial import distance as dist
import numpy as np
import cv2
import dlib
import pyautogui
import collections

FACIAL_LANDMARKS_IDXS = collections.OrderedDict([("mouth", (48, 68)),("right_eyebrow", (17, 22)),("left_eyebrow", (22, 27)),("right_eye", (36, 42)),("left_eye", (42, 48)),("nose", (27, 35)),("jaw", (0, 17))])
EYE_AR_THRESH = 0.3
EYE_AR_COUNTER = 6
LEYE_AR_THRESH = 0.2
REYE_AR_THRESH = 0.2
EYE_DIFF_THRESH = 0.01
MOUTH_AR_THRESH=0.2
MOUTH_AR_COUNTER=4
LECOUNTER=0
RECOUNTER=0
MCOUNTER=0
TOTAL=0
SCROLL=False
MOUSEMOVE=-1
Anchor=(0,0)
cap = cv2.VideoCapture(0)
window = "Frame"

trainingpath = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(trainingpath)
detector = dlib.get_frontal_face_detector()

def ear(eye):     #Eye Aspect Ratio
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    
    ear = (A+B)/ (2*C)
    return ear

def mouthar(mouth):       #Mouth Aspect Ratio
    mouth=mouth[-8:]
    A=dist.euclidean(mouth[1],mouth[7])
    B=dist.euclidean(mouth[2],mouth[6])
    C=dist.euclidean(mouth[3],mouth[5])
    D=dist.euclidean(mouth[0],mouth[4])
    
    mar=(A+B+C)/(3*D)
    return mar

def noself(anchor,current,up=False,left=False):
    nx = int(np.reshape(current,(2,1))[0])
    ny = int(np.reshape(current,(2,1))[1])
    x = int(np.reshape(anchor,(2,1))[0])
    y = int(np.reshape(anchor,(2,1))[1])
    
    cv2.line(frame,(x,y),(nx,ny),(255,0,0),2)
    
    if nx>x+40 and left:
        return "Right"
    elif nx<x-40 and left:
        return "Left"
    
    if ny>y+20 and up:
        return "Down"
    elif ny<y-20 and up:
        return "Up"

    return "--"

def get_landmarks(frame):
    rects = detector(frame,0)
    return np.matrix([[p.x, p.y] for p in predictor(frame,rects[0]).parts()])


def getshape(organ,landmarks):
    (i,j) = FACIAL_LANDMARKS_IDXS[organ]
    return landmarks[i:j]

while(cap.isOpened()):
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    landmarks = get_landmarks(frame)
    
    lshape = getshape("left_eye",landmarks)
    rshape = getshape("right_eye",landmarks)
    
    temp=lshape
    lshape=rshape
    rshape=temp
    
    mouthshape = getshape("mouth",landmarks)
    noseshape = getshape("nose",landmarks)
    lear = ear( lshape )
    rear = ear( rshape )
    diff_ear = np.abs(lear-rear)
    finalear = (lear+rear)/2
    mar=mouthar(mouthshape)
    cv2.putText(frame,"l Ratio = " + str(lear),(10,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(frame,"R Ratio="+str(rear),(10,60),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,0,255),2)
#    cv2.putText(frame,"fear Ratio="+str(finalear),(10,90),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,0,255),2)
#    cv2.putText(frame,"Total Blinks = " + str(TOTAL),(10,300),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,0,255),2)
    leftEyeHull = cv2.convexHull(lshape)
    rightEyeHull = cv2.convexHull(rshape)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    
    if diff_ear > EYE_DIFF_THRESH:
    
        if lear<rear:
            if lear<EYE_AR_THRESH:
                LECOUNTER+=1
            else:
                if LECOUNTER>=EYE_AR_COUNTER:
                    print("Left Click")
                    pyautogui.click(button='left')
                    TOTAL+=1
                LECOUNTER=0
        else:
            if rear<EYE_AR_THRESH:
                RECOUNTER+=1
            else:
                if RECOUNTER>=EYE_AR_COUNTER:
                    print("Right Click")
                    pyautogui.click(button='right')
                    TOTAL+=1
                RECOUNTER=0
    
    if mar>MOUTH_AR_THRESH:
        MCOUNTER+=1
    else:
        if MCOUNTER>=MOUTH_AR_COUNTER:
            if MOUSEMOVE==1:
                MOUSEMOVE=0
                Anchor = noseshape[4]
            else:
                MOUSEMOVE=1
                Anchor = noseshape[4]
            
        MCOUNTER=0;
        
    if MOUSEMOVE==1:
        cv2.putText(frame,"Mouse Move mode:"+str(MOUSEMOVE),(10,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,0,255),2)
        direc=noself(Anchor,noseshape[3],up=True,left=True)
        cv2.putText(frame,direc,(10,60),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,0,255),2)
        if direc=='Left' or direc=='Right':
            if direc=='Left':
                pyautogui.moveRel(-20,0)
            else:
                pyautogui.moveRel(20,0)
        elif direc=='Up' or direc=='Down':
            if direc=='Up':
                pyautogui.moveRel(0,-20)
            else:
                pyautogui.moveRel(0,20)
                
    if MOUSEMOVE==0:
        cv2.putText(frame,"Scroll mode:"+str(MOUSEMOVE),(10,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,0,255),2)
        direc=noself(Anchor,noseshape[3],up=True,left=True)
        cv2.putText(frame,direc,(10,60),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,0,255),2)
        if direc=='Up' or direc=='Down':
            if direc=='Up':
                pyautogui.scroll(50)
            else:
                pyautogui.scroll(-50)
    
    cv2.imshow(window,frame)
    if cv2.waitKey(1)==27:
        break;

cv2.destroyAllWindows()
cap.release()

    
    