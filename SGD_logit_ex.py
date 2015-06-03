import numpy as np
from sklearn.linear_model import SGDClassifier
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])
clf = SGDClassifier()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

data = np.loadtxt("data/zoo.csv", delimiter=",")  
zoo_target = data[:,-1]
zoo_data = data[:,:data.shape[1]-1]
logreg = SGDClassifier(loss="log")
logreg.fit(zoo_data, zoo_target)
print logreg.predict([[1,0,1,0,0,1,0,1,1,0,0,1,1,1,0]])