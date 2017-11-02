from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# this is a random generated distribution,using make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)
#build rf
clf = RandomForestClassifier(max_depth=2, random_state=0)
#fit
clf.fit(X, y)
#predict
print(clf.predict([[0, 0, 0, 0]]))
