import cv2
import numpy as np

import matplotlib.pyplot as plt
from sklearn import neighbors,svm,metrics

from sklearn.model_selection import train_test_split
from mnist import MNIST
from sklearn.externals import joblib
def findRoi(frame, thresValue, margin):  
    rois = []  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    gray2 = cv2.dilate(gray,None,iterations=2)  
    gray2 = cv2.erode(gray2,None,iterations=2)  
    edges = cv2.absdiff(gray,gray2)  
    x = cv2.Sobel(edges,cv2.CV_16S,1,0)  
    y = cv2.Sobel(edges,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)  
    absY = cv2.convertScaleAbs(y)  
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)  
    ret, ddst = cv2.threshold(dst,thresValue,255,cv2.THRESH_BINARY)  
    im, contours, hierarchy = cv2.findContours(ddst,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    for c in contours:  
        x, y, w, h = cv2.boundingRect(c)
        if w > 15 and h > 20:
            if x-margin>=0:
                x=x-margin
                w+=margin
            else:
                x=0
                w+=x
            if y-margin>=0:
                y=y-margin
                h+=margin
            else:
                y=0
                h+=y
            if x+w+margin<frame.shape[1]:
                w+=margin
            if y+h+margin<frame.shape[0]:
                h+=margin
            rois.append((x,y,w,h))
        
    return rois, edges

def findDigit(clf, roi, thresValue,size_x,size_y):  
    ret, th = cv2.threshold(roi, thresValue, 255, cv2.THRESH_BINARY)  
    th = cv2.resize(th,(size_x,size_y))  
    out = th.reshape(-1,size_x*size_y).astype(np.float32)
    out=out/255
    if clf is not None:
        ret=clf.predict(out) 
    return ret,th

def concatenate(images,num_images_per_coloum):  
    n = len(images)
    p=int(np.ceil(n/num_images_per_coloum))
    size_x=images[0].shape[1]
    size_y=images[0].shape[0]
    output=np.zeros((num_images_per_coloum*size_y,size_x*p))
    #output = np.zeros(size_x*size_y*n).reshape(-1,size_x)  
    for i in range(n):
        c=int(i/num_images_per_coloum)
        m=i%10
        output[size_y*m:size_y*(m+1),size_x*c:size_x*(c+1)] = images[i]  
    return output

def detectPicture(image,clf=None,size_x=20,size_y=20,single_flag=1):
    tmp=image.copy()
    rois, edges = findRoi(tmp, 50, 2)
    digits = [] 
    for r in rois:
        x, y, w, h = r  
        digit, th = findDigit(clf, edges[y:y+h,x:x+w], 30, size_x,size_y)  
        digits.append(th)  
        cv2.rectangle(tmp, (x,y), (x+w,y+h), (153,153,0), 2)  
        cv2.putText(tmp, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)
    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    cv2.imshow('pic',tmp)
    Nd = len(digits)
    if Nd>0:
        output = concatenate(digits,10)  
        cv2.imshow('digits', output)
    if single_flag:
        cv2.waitKey(0)
    return digits,edges,output

def getDataAndLabelFromPic(img,size_x=20,size_y=20):
    tmp=img.copy()
    digits,edges,conc=detectPicture(img,size_x=size_x,size_y=size_y)
    Nd = len(digits)
    cnt=0
    label=[]
    while cnt<Nd:
        tmp_conc=conc.copy()
        x=int(cnt/10)*size_x
        y=(cnt%10)*size_y
        cv2.rectangle(tmp_conc, (x,y), (x+size_x,y+size_y), (153,153,0), 2)
        cv2.imshow('digits',tmp_conc)
        cv2.waitKey(1)
        print('input the digits(separate by space),%d numbers left:'%(Nd-cnt))  
        numbers = input().split(' ')
        Nn = len(numbers)
        cnt += Nn
##        if Nd != Nn:  
##            print('Not equal,required number %d, input %d,update CLF fail!'%(Nd,Nn))  
        try:  
            for i in range(Nn):  
                numbers[i] = int(numbers[i])  
        except:  
            print('error')
        label = np.hstack((label,numbers))
    if len(label)!=Nd:
        print('Not equal,required number %d, input %d,update CLF fail!'%(Nd,len(label)))
    newData=np.array(digits).reshape(-1,size_x*size_y).astype(np.float32)
    newData/=255
    return newData,np.array(label)
