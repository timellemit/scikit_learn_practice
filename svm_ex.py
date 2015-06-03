from sklearn import svm
from sklearn.metrics import classification_report
from numpy import loadtxt
data = loadtxt("zoo.txt", delimiter=",")
nrow, ncol = data.shape
X_train = data[0:nrow/2-1,0:ncol-2]
y_train = data[0:nrow/2-1,ncol-1]
X_test = data[nrow/2:nrow,0:ncol-2]
y_test = data[nrow/2:nrow:,ncol-1]
clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print "Accuracy: " + str(clf.score(X_test, y_test))
print classification_report(y_test, pred)


