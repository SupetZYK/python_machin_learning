import cv2
import numpy as np

#这是总共5000个数据，0-9各500个，我们读入图片后整理数据，
#这样得到的train和trainLabel依次对应，图像数据和标签。
def initKnn():  
    knn = cv2.ml.KNearest_create()  
    img = cv2.imread('digits.png')  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]  
    train = np.array(cells).reshape(-1,400).astype(np.float32)  
    trainLabel = np.repeat(np.arange(10),500)  
    return knn, train, trainLabel

#updateKnn是增加自己的训练数据后更新Knn的操作。
def updateKnn(knn, train, trainLabel, newData=None, newDataLabel=None):  
    if newData is not None and newDataLabel is not None:
        print(train.shape, newData.shape)  
        newData = newData.reshape(-1,400).astype(np.float32)  
        train = np.vstack((train,newData))  
        trainLabel = np.hstack((trainLabel,newDataLabel))  
    knn.train(train,cv2.ml.ROW_SAMPLE,trainLabel)  
    return knn, train, trainLabel

#findRoi函数是找到每个数字的位置，
#用包裹其最小矩形的左上顶点的坐标和该矩形长宽表示(x, y, w, h)。
#这里还用到了Sobel算子。edges是原始图像形态变换之后的灰度图，
#可以排除一些背景的影响，比如本子边缘、纸面的格子、手、笔以及影子等等，
#用edges来获取数字图像效果比Sobel获取的边界效果要好。
def findRoi(frame, thresValue):  
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
        if w > 10 and h > 20:  
            rois.append((x,y,w,h))  
    return rois, edges
#findDigit函数是用KNN来分类，
#并将结果返回。th是用来手动输入训练数据时显示的图片。
#20x20pixel的尺寸是OpenCV自带digits.png中图像尺寸，
#因为我是在其基础上更新数据，所以沿用这个尺寸。
def findDigit(knn, roi, thresValue):  
    ret, th = cv2.threshold(roi, thresValue, 255, cv2.THRESH_BINARY)
    #th=roi
    th = cv2.resize(th,(20,20))  
    out = th.reshape(-1,400).astype(np.float32)  
    ret, result, neighbours, dist = knn.findNearest(out, k=5)  
    return int(result[0][0]), th
#concatenate函数是拼接数字图像并显示的，用来输入训练数据。
def concatenate(images):  
    n = len(images)  
    output = np.zeros(20*20*n).reshape(-1,20)  
    for i in range(n):  
        output[20*i:20*(i+1),:] = images[i]  
    return output



knn, train, trainLabel = initKnn()
knn, train, trainLabel = updateKnn(knn, train, trainLabel)
cap = cv2.VideoCapture(0)
#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = 426*2
height = 480
#videoFrame = cv2.VideoWriter('frame.avi',cv2.VideoWriter_fourcc('M','J','P','G'),25,(int(width),int(height)),True)
count = 0
while True:  
    ret, frame = cap.read()  
    frame = frame[:,:426]  
    rois, edges = findRoi(frame, 50)  
    digits = []  
    for r in rois:  
        x, y, w, h = r  
        digit, th = findDigit(knn, edges[y:y+h,x:x+w], 20)  
        digits.append(cv2.resize(th,(20,20)))  
        cv2.rectangle(frame, (x,y), (x+w,y+h), (153,153,0), 2)  
        cv2.putText(frame, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)  
    newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  
    newFrame = np.hstack((frame,newEdges))  
    cv2.imshow('frame', newFrame)
    #videoFrame.write(newFrame) 
    key = cv2.waitKey(1) & 0xff  
    if key == ord(' '):  
        break  
    elif key == ord('x'):  
        Nd = len(digits)  
        output = concatenate(digits)  
        showDigits = cv2.resize(output,(60,60*Nd))  
        cv2.imshow('digits', showDigits)  
        #cv2.imwrite(str(count)+'.png', showDigits)  
        count += 1  
        if cv2.waitKey(0) & 0xff == ord('e'):  
            pass  
        print('input the digits(separate by space):')  
        numbers = input().split(' ')  
        Nn = len(numbers)  
        if Nd != Nn:  
            print('update KNN fail!')  
            continue  
        try:  
            for i in range(Nn):  
                numbers[i] = int(numbers[i])  
        except:  
            continue  
        knn, train, trainLabel = updateKnn(knn, train, trainLabel, output, numbers)  
        print('update KNN, Done!')

print('Numbers of trained images:',len(train))
print('Numbers of trained image labels', len(trainLabel))
cap.release()
cv2.destroyAllWindows()
