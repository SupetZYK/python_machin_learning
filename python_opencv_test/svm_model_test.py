import cv2
import numpy as np

import matplotlib.pyplot as plt
from sklearn import neighbors,svm,metrics
from sklearn.externals import joblib
from mnist import MNIST
clf = joblib.load("digits_svm_model.m")

# #OpenCV dataset
##img = cv2.imread('digits.png')  
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
##cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
##data=np.array(cells).astype(np.float32)
##label = np.repeat(np.arange(10),500)
##X=None
##for i in range(data.shape[0]):
##    for j in range(data.shape[1]):
##        if X is None:
##            X=cv2.resize(data[i,j],(28,28)).reshape(1,-1)
##        else:
##            X=np.vstack((X,cv2.resize(data[i,j],(28,28)).reshape(1,-1)))
##X=X/255
##predicted=clf.predict(X)
##print("Classification report for classifier %s:\n%s\n"
##      % (clf, metrics.classification_report(label, predicted)))
##print("Confusion matrix:\n%s" % metrics.confusion_matrix(label, predicted))

# #MNIST dataset
mndata = MNIST('.')
mndata.load_testing()
X_test=np.array(mndata.test_images).astype(np.float32)
X_test=X_test/255
y_test=np.array(mndata.test_labels)
cv2.imshow('a',X_test[0].reshape(28,28))
##img=cv2.imread('testNumber.jpg')
##img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##img=img.astype(np.float32)
##img = cv2.erode(img,None,iterations=5) 
##img=cv2.resize(img,(28,28))
##ret, th = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
##
##x=(255-th)/255
##print(clf.predict(x.reshape(1,-1)))
