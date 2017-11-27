from digits_lib import *
# #classifier
clf=neighbors.KNeighborsClassifier()
##clf=svm.SVC(gamma=0.01)
# load original dataset
img = cv2.imread('digits.png')  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]  
data = np.array(cells).reshape(-1,400).astype(np.float32)
data=data/255
label = np.repeat(np.arange(10),500)
#clf.fit(data,label)

img1=cv2.imread('zyk_test.jpg')
newData,newLabl=getDataAndLabelFromPic(img1)
data=np.vstack((data,newData))
label=np.hstack((label,newLabl))
