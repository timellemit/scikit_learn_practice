import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
#print iris.data.shape, iris.target.shape

# just one validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0)

#print X_train.shape, y_train.shape
#print X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print "Iris. One-validation accuracy: " + str(clf.score(X_test, y_test))  

# iris 5-fold cross-validation
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print("Iris. 5-fold CV accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))        

zoo = np.loadtxt("zoo.txt", delimiter=",")  
zoo_data = zoo[:,0:zoo.shape[1]-2]
zoo_target = zoo[:,zoo.shape[1]-1] 
# zoo 5-fold cross-validation
clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, zoo_data, zoo_target, cv=5)
print("Zoo. 5-fold CV accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))     
                            