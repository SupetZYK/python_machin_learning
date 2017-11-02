from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
# image process library
from PIL import Image
import numpy as np
#dataset
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#build rf
clf = RandomForestClassifier(n_estimators=15,max_depth=30, random_state=0)
#fit
clf.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = clf.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# load image
img=Image.open('testNumber.jpg')
img=img.convert('L')
img=img.resize((8,8),Image.ANTIALIAS)
im=np.array(img)

print(clf.predict(im.reshape(1,-1)))
plt.imshow(im,cmap='gray')
plt.show()
