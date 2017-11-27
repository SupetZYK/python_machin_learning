from digits_lib import *

#classifier
#clf=neighbors.KNeighborsClassifier()
clf=svm.SVC(gamma=0.01)
# load dataset
if 0:
    img = cv2.imread('digits.png')  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]  
    data = np.array(cells).reshape(-1,400).astype(np.float32)  
    label = np.repeat(np.arange(10),500)

##    # split into train and test
##    X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.3,stratify=label)
##    X_train=X_train/255
##    X_test=X_test/255
##    #fit or load model
##    clf.fit(X_train,y_train)

    data=data/255
    clf.fit(data,label)
    size_x=20
    size_y=20

    # user train image
    img=cv2.imread('train_cv_img.jpg')
    rois, edges = findRoi(img, 30, 2)
    digits = [] 
    for r in rois:  
        x, y, w, h = r  
        digit, th = findDigit(clf, edges[y:y+h,x:x+w], 30, size_x,size_y)  
        digits.append(cv2.resize(th,(size_x,size_y)))  
        cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 2)  
        cv2.putText(img, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)
    cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    cv2.imshow('a',img)
    
    Nd = len(digits)
    for i in range(int(np.ceil(Nd/8))):
        start_idx=i*8
        end_idx=(i+1)*8
        if end_idx>Nd:
            end_idx=Nd
        output = concatenate(digits[start_idx:end_idx])  
        showDigits = cv2.resize(output,(30,30*(end_idx-start_idx)))
        cv2.namedWindow("digits", cv2.WINDOW_NORMAL)
        cv2.imshow('digits', showDigits)
        if cv2.waitKey(0) & 0xff == ord('x'):
            pass
        print('input the digits(separate by space):')  
        numbers = input().split(' ')  
        Nn = len(numbers)  
        if (end_idx-start_idx) != Nn:  
            print('update CLF fail!')  
        try:  
            for i in range(Nn):  
                numbers[i] = int(numbers[i])  
        except:  
            print('error')
        newData=output.reshape(-1,size_x*size_y).astype(np.float32)
        newData/=255
        data = np.vstack((data,newData))
        label = np.hstack((label,numbers))
    clf.fit(data,label)
    print('update CLF succusee!')  
            
else:
    clf = joblib.load("clf11_16")
##    mndata = MNIST('.')
##    mndata.load_training()
##    mndata.load_testing()
##    X_train=np.array(mndata.train_images).astype(np.float32)
##    X_train=X_train/255
##    y_train=np.array(mndata.train_labels)
##    X_test=np.array(mndata.test_images).astype(np.float32)
##    X_test=X_test/255
##    y_test=np.array(mndata.test_labels)
    size_x=20
    size_y=20

#predict
#predicted=clf.predict(X_test)
#print("Classification report for classifier %s:\n%s\n"
      #% (clf, metrics.classification_report(y_test, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))


cap = cv2.VideoCapture(0)
width = 426*2
height = 480
#videoFrame = cv2.VideoWriter('frame.avi',cv2.VideoWriter_fourcc('M','J','P','G'),25,(int(width),int(height)),True)

count = 0
while True:  
    ret, frame = cap.read()  
    frame = frame[:,:426]  
    digits,edges=detectPicture(frame,clf,20,20,0)
    newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  
    newFrame = np.hstack((frame,newEdges))  
    cv2.imshow('frame', newFrame)  
    #videoFrame.write(newFrame)  
    key = cv2.waitKey(1) & 0xff  
    if key == ord(' '):  
        break
    elif key == ord('x'):  
        Nd = len(digits)  
        output = concatenate(digits,10)  
        #showDigits = cv2.resize(output,(60,60*Nd))  
        cv2.imshow('digits', output)
        if cv2.waitKey(0) & 0xff == ord('e'):  
            pass
cap.release()
cv2.destroyAllWindows()

